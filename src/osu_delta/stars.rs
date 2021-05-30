use super::{
    difficulty_range_ar, difficulty_range_od, math_util::logistic, DifficultyObject, Movement,
    OsuObject, Skill, SkillKind, SliderState,
};

use rosu_pp::{osu::DifficultyAttributes, Beatmap, Mods, StarResult};
use std::collections::VecDeque;

const OBJECT_RADIUS: f32 = 64.0;
const SECTION_LEN: f32 = 400.0;
const DIFFICULTY_MULTIPLIER: f32 = 0.0675;
const NORMALIZED_RADIUS: f32 = 52.0;

/// Star calculation for osu!standard maps.
///
/// Slider paths are considered but stack leniency is ignored.
/// As most maps don't even make use of leniency and even if,
/// it has generally little effect on stars, the results are close to perfect.
///
/// In case of a partial play, e.g. a fail, one can specify the amount of passed objects.
pub fn stars(map: &Beatmap, mods: impl Mods, passed_objects: Option<usize>) -> StarResult {
    let take = passed_objects.unwrap_or_else(|| map.hit_objects.len());

    let map_attributes = map.attributes().mods(mods);
    let hitwindow = difficulty_range_od(map_attributes.od).floor() / map_attributes.clock_rate;
    let od = (80.0 - hitwindow) / 6.0;

    let mut diff_attributes = DifficultyAttributes {
        ar: map_attributes.ar,
        od,
        ..Default::default()
    };

    if take < 2 {
        return StarResult::Osu(diff_attributes);
    }

    let radius = OBJECT_RADIUS * (1.0 - 0.7 * (map_attributes.cs - 5.0) / 5.0) / 2.0;
    let mut scaling_factor = NORMALIZED_RADIUS / radius;

    if radius < 30.0 {
        let small_circle_bonus = (30.0 - radius).min(5.0) / 50.0;
        scaling_factor *= 1.0 + small_circle_bonus;
    }

    let mut slider_state = SliderState::new(map);
    let mut ticks_buf = Vec::new();

    let hit_objects_iter = map.hit_objects.iter().take(take).filter_map(|h| {
        OsuObject::new(
            h,
            map,
            radius,
            scaling_factor,
            &mut ticks_buf,
            &mut diff_attributes,
            &mut slider_state,
        )
    });

    let mut hit_objects = Vec::with_capacity(take);
    hit_objects.extend(hit_objects_iter);

    let map_len = (hit_objects.last().unwrap().time - hit_objects[0].time)
        / 1000.0
        / map_attributes.clock_rate;

    let preempt_no_clock_rate = difficulty_range_ar(map.ar);

    let note_densities = note_density(&hit_objects, preempt_no_clock_rate);
    let tap_attributes = calculate_tap_attributes(&hit_objects, map_attributes.clock_rate, radius);
    let finger_control_diff =
        calculate_finger_control_diff(&hit_objects, map_attributes.clock_rate);

    todo!()
}

const PROB_TRESHOLD: f32 = 0.02;
const TIME_THRESHOLD_BASE: f32 = 1200.0;
const TP_MIN: f32 = 0.1;
const TP_MAX: f32 = 100.0;
const PROB_PRECISION: f32 = 1e-4;
const TIME_PRECISION: f32 = 0.6;
const MAX_ITERATIONS: usize = 100;
const DEFAULT_CHEESE_LEVEL: f32 = 0.4;
const CHEESE_LEVEL_COUNT: usize = 11;
const MISS_TP_COUNT: usize = 20;
const COMBO_TP_COUNT: usize = 50;

fn calculate_aim_attributes(
    hit_objects: &[OsuObject],
    clock_rate: f32,
    strain_history: &[[f32; 4]],
    note_densities: &[f32],
) -> AimAttributes {
    todo!()
}

struct AimAttributes;

fn create_movements(
    hit_objects: &[OsuObject],
    clock_rate: f32,
    strain_history: &[[f32; 4]],
    hidden: bool,
    note_densities: Option<Vec<f32>>,
) -> Vec<Movement> {
    todo!()
}

fn calculate_finger_control_diff(hit_objects: &[OsuObject], clock_rate: f32) -> f32 {
    let mut prev_time = hit_objects[0].time / 1000.0;
    let mut curr_strain = 0.0;
    let mut prev_strain_time = 0.0;
    let mut repeat_strain_count = 1;
    let mut strain_history = Vec::new();
    strain_history.push(0.0);

    for i in 1..hit_objects.len() {
        let curr_time = hit_objects[i].time / 1000.0;
        let delta_time = (curr_time - prev_time) / clock_rate;
        let strain_time = delta_time.max(0.046875);
        let strain_decay_base = 0.9_f32.powf(strain_time.min(0.2).recip());

        curr_strain *= strain_decay_base.powf(delta_time);
        strain_history.push(curr_strain);
        let mut strain = 0.1 / strain_time;
        if (strain_time - prev_strain_time).abs() > 0.004 {
            repeat_strain_count = 1;
        } else {
            repeat_strain_count += 1;
        }

        if hit_objects[i].is_slider() {
            strain /= 2.0;
        }

        if repeat_strain_count % 2 == 0 {
            strain = 0.0;
        } else {
            strain /= 1.25_f32.powi(repeat_strain_count);
        }

        curr_strain += strain;
        prev_time = curr_time;
        prev_strain_time = strain_time;
    }

    strain_history.sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

    let mut diff = 0.0;
    let k: f32 = 0.95;

    for i in 0..hit_objects.len() {
        diff += strain_history[i] * k.powi(i as i32);
    }

    diff * (1.0 - k) * 1.1
}

fn calculate_tap_attributes(
    hit_objects: &[OsuObject],
    clock_rate: f32,
    radius: f32,
) -> TapAttributes {
    let (strain_history, tap_diff) = calculate_tap_strain(hit_objects, 0.0, clock_rate, radius);
    let burst_strain = strain_history
        .iter()
        .map(|s| s[0])
        .max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .expect("no burst strain");

    let streamness_mask = calculate_streamness_mask(hit_objects, burst_strain, clock_rate);

    let stream_note_count = streamness_mask.iter().sum();

    let (_, mash_tap_diff) = calculate_tap_strain(hit_objects, 1.0, clock_rate, radius);

    TapAttributes {
        difficulty: tap_diff,
        stream_note_count,
        mashed_difficulty: mash_tap_diff,
        strain_history,
    }
}

struct TapAttributes {
    difficulty: f32,
    stream_note_count: f32,
    mashed_difficulty: f32,
    strain_history: Vec<[f32; 4]>,
}

fn calculate_streamness_mask(hit_objects: &[OsuObject], skill: f32, clock_rate: f32) -> Vec<f32> {
    let mut streamness_mask = vec![0.0; hit_objects.len()];

    let stream_time_threshold = skill.powf(-2.7 / 3.2);

    for i in 1..hit_objects.len() {
        let t = (hit_objects[i].time - hit_objects[i - 1].time) / 1000.0 / clock_rate;
        streamness_mask[i] = 1.0 - logistic((t / stream_time_threshold - 1.0) * 15.0);
    }

    streamness_mask
}

// Four elements, evenly spaces between 2.3 and -2.8,
// then pointwise applied the exp function on
const DECAY_COEFFS: [f32; 4] = [9.97418, 1.82212, 0.332871, 0.0608101];

trait ArrayVec: Sized {
    fn neg(&mut self) -> &mut Self;
    fn scalar_mult(&mut self, scalar: f32) -> &mut Self;
    fn pointwise_exp(&mut self) -> &mut Self;
    fn powi_mean(&self, pow: i32) -> f32;
    fn pointwise_add(&mut self, other: &Self) -> &mut Self;
    fn pointwise_mult(&mut self, other: &Self) -> &mut Self;
    fn pointwise_powf(&mut self, pow: f32) -> &mut Self;
}

impl ArrayVec for [f32; 4] {
    fn neg(&mut self) -> &mut Self {
        for elem in self.iter_mut() {
            *elem *= -1.0;
        }

        self
    }

    fn scalar_mult(&mut self, scalar: f32) -> &mut Self {
        for elem in self.iter_mut() {
            *elem *= scalar;
        }

        self
    }

    fn pointwise_exp(&mut self) -> &mut Self {
        for elem in self.iter_mut() {
            *elem = elem.exp();
        }

        self
    }

    fn powi_mean(&self, pow: i32) -> f32 {
        let mut sum = 0.0;

        for elem in self.iter() {
            sum += elem.powi(pow);
        }

        (sum / self.len() as f32).powf(1.0 / pow as f32)
    }

    fn pointwise_add(&mut self, other: &Self) -> &mut Self {
        for (elem, term) in self.iter_mut().zip(other.iter()) {
            *elem += term;
        }

        self
    }

    fn pointwise_mult(&mut self, other: &Self) -> &mut Self {
        for (elem, factor) in self.iter_mut().zip(other.iter()) {
            *elem *= factor;
        }

        self
    }

    fn pointwise_powf(&mut self, pow: f32) -> &mut Self {
        for elem in self.iter_mut() {
            *elem = elem.powf(pow);
        }

        self
    }
}

use std::cmp::Ordering;

const SPACED_BUFF_FACTOR: f32 = 0.1;
const TIMESCALE_FACTOR: [f32; 4] = [1.02, 1.02, 1.05, 1.15];

fn calculate_tap_strain(
    hit_objects: &[OsuObject],
    mash_level: f32,
    clock_rate: f32,
    radius: f32,
) -> (Vec<[f32; 4]>, f32) {
    let mut strain_history = vec![[0.0; 4]; 2];
    let mut curr_strain = [0.0; 4];

    let mut prev_prev_time = hit_objects[0].time / 1000.0;
    let mut prev_time = hit_objects[1].time / 1000.0;

    for i in 2..hit_objects.len() {
        let curr_time = hit_objects[i].time / 1000.0;

        curr_strain.pointwise_mult(
            DECAY_COEFFS
                .clone()
                .neg()
                .scalar_mult((curr_time - prev_time) / clock_rate)
                .pointwise_exp(),
        );

        let mut strain = curr_strain.clone();
        strain.pointwise_powf(1.1 / 3.0 * 1.5);

        strain_history.push(strain);

        let dist = (hit_objects[i].pos - hit_objects[i - 1].pos).length() / (2.0 * radius);

        let spaced_buf = calculate_spacedness(dist) * SPACED_BUFF_FACTOR;

        let delta_time = ((curr_time - prev_time) / clock_rate).max(0.01);

        let strain_addition = (delta_time.powf(-2.7) * 0.265).max(delta_time.sqrt().recip());

        curr_strain.pointwise_add(DECAY_COEFFS.clone().scalar_mult(
            strain_addition
                * calculate_mash_nerf_factor(dist, mash_level).powi(3)
                * (1.0 + spaced_buf).powi(3),
        ));

        prev_prev_time = prev_time;
        prev_time = curr_time;
    }

    let mut strain_result = [0.0; 4];

    for j in 0..4 {
        let mut single_strain_history = vec![0.0; hit_objects.len()];

        for i in 0..hit_objects.len() {
            single_strain_history[i] = strain_history[i][j];
        }

        single_strain_history.sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

        let mut single_strain_result = 0.0;
        let k = 1.0 - 0.04 * DECAY_COEFFS[j].sqrt();

        for i in 0..hit_objects.len() {
            single_strain_result += single_strain_history[i] * k.powi(i as i32);
        }

        strain_result[j] = single_strain_result * (1.0 - k) * TIMESCALE_FACTOR[j];
    }

    let diff = strain_result.powi_mean(2);

    (strain_history, diff)
}

#[inline]
fn calculate_mash_nerf_factor(relative_d: f32, mash_level: f32) -> f32 {
    let full_mash_factor = 0.73 + 0.27 * logistic(relative_d * 7.0 - 6.0);

    mash_level * full_mash_factor + (1.0 - mash_level)
}

#[inline]
fn calculate_spacedness(d: f32) -> f32 {
    logistic((d - 0.533) / 0.13) - logistic(-4.1)
}

fn note_density(hit_objects: &[OsuObject], preempt: f32) -> Vec<f32> {
    let mut note_densities = Vec::new();
    let mut window = VecDeque::new();

    let mut next = 0;

    for time in hit_objects.iter().map(|h| h.time) {
        while next < hit_objects.len() && hit_objects[next].time < time + preempt {
            window.push_front(&hit_objects[next]);
            next += 1;
        }

        while window
            .back()
            .filter(|back| back.time < time - preempt)
            .is_some()
        {
            window.pop_back();
        }

        let density = window
            .iter()
            .map(|h| 1.0 - (h.time - time).abs() / preempt)
            .sum();

        note_densities.push(density);
    }

    note_densities
}
