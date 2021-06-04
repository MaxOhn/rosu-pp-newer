use crate::osu_delta::math_util::try_expand_find_root_brent;

use super::math_util::{logistic, try_find_root_bisection, try_find_root_brent, PoissonBinomial};
use super::{
    difficulty_range_ar, difficulty_range_od, ArrayVec, DifficultyAttributes, HitProbabilities,
    Movement, OsuObject, SliderState,
};

use rosu_pp::{Beatmap, Mods};
use std::cmp::Ordering;
use std::collections::VecDeque;

const OBJECT_RADIUS: f32 = 64.0;

const AIM_MULTIPLIER: f32 = 0.641;
const TAP_MULTIPLIER: f32 = 0.641;
const FINGER_CONTROL_MULTIPLIER: f32 = 1.245;
const SR_EXPONENT: f32 = 0.83;

/// Star calculation for osu!standard maps.
///
/// Slider paths are considered but stack leniency is ignored.
/// As most maps don't even make use of leniency and even if,
/// it has generally little effect on stars, the results are close to perfect.
///
/// In case of a partial play, e.g. a fail, one can specify the amount of passed objects.
pub fn stars(
    map: &Beatmap,
    mods: impl Mods,
    passed_objects: Option<usize>,
) -> DifficultyAttributes {
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
        return diff_attributes;
    }

    let radius = OBJECT_RADIUS * (1.0 - 0.7 * (map_attributes.cs - 5.0) / 5.0) / 2.0;

    let mut slider_state = SliderState::new(map);
    let mut ticks_buf = Vec::new();

    let hit_objects_iter = map.hit_objects.iter().take(take).filter_map(|h| {
        OsuObject::new(
            h,
            map,
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

    let preempt_no_clock_rate = difficulty_range_ar(map_attributes.ar);
    let note_densities = note_density(&hit_objects, preempt_no_clock_rate);
    let tap_attributes = calculate_tap_attributes(&hit_objects, map_attributes.clock_rate, radius);
    let finger_control_diff =
        calculate_finger_control_diff(&hit_objects, map_attributes.clock_rate);
    let aim_attributes = calculate_aim_attributes(
        &hit_objects,
        map_attributes.clock_rate,
        &tap_attributes.strain_history,
        &note_densities,
        radius,
        mods.hd(),
    );

    let tap_sr = TAP_MULTIPLIER * tap_attributes.difficulty.powf(SR_EXPONENT);
    let aim_sr = AIM_MULTIPLIER * aim_attributes.fc_prob_throughput.powf(SR_EXPONENT);
    let finger_control_sr = FINGER_CONTROL_MULTIPLIER * finger_control_diff.powf(SR_EXPONENT);

    let sr = [tap_sr, aim_sr, finger_control_sr].powi_mean(7) * 1.131;

    diff_attributes.stars = sr;
    diff_attributes.length = map_len;

    diff_attributes.tap_sr = tap_sr;
    diff_attributes.tap_diff = tap_attributes.difficulty;
    diff_attributes.stream_note_count = tap_attributes.stream_note_count;
    diff_attributes.mash_tap_diff = tap_attributes.mashed_difficulty;

    diff_attributes.finger_control_sr = finger_control_sr;
    diff_attributes.finger_control_diff = finger_control_diff;

    diff_attributes.aim_sr = aim_sr;
    diff_attributes.aim_diff = aim_attributes.fc_prob_throughput;
    diff_attributes.aim_hidden_factor = aim_attributes.hidden_factor;
    diff_attributes.combo_tps = aim_attributes.combo_throughputs;
    diff_attributes.miss_tps = aim_attributes.miss_throughputs;
    diff_attributes.miss_counts = aim_attributes.miss_counts;
    diff_attributes.cheese_note_count = aim_attributes.cheese_note_count;
    diff_attributes.cheese_levels = aim_attributes.cheese_levels;
    diff_attributes.cheese_factors = aim_attributes.cheese_factor;

    diff_attributes.n_circles = map.n_circles;
    diff_attributes.n_sliders = map.n_sliders;
    diff_attributes.n_spinners = map.n_spinners;

    diff_attributes
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
    radius: f32,
    hidden: bool,
) -> AimAttributes {
    let movements = create_movements(
        hit_objects,
        clock_rate,
        strain_history,
        false,
        note_densities,
        radius,
    );

    let mut combo_section_amount = COMBO_TP_COUNT;

    if movements.len() < combo_section_amount {
        combo_section_amount = movements.len();
    }

    let mut miss_section_amount = MISS_TP_COUNT;

    if movements.len() < miss_section_amount {
        miss_section_amount = movements.len();
    }

    if movements.is_empty() {
        return AimAttributes::default();
    }

    let mut map_hit_probs =
        HitProbabilities::new(&movements, DEFAULT_CHEESE_LEVEL, combo_section_amount);
    let fc_prob_tp = calculate_fc_prob_tp(&movements, DEFAULT_CHEESE_LEVEL);

    let hidden_factor = if hidden {
        let movement_hidden = create_movements(
            hit_objects,
            clock_rate,
            strain_history,
            true,
            note_densities,
            radius,
        );

        let fc_prob_tp_hidden = calculate_fc_prob_tp(&movement_hidden, DEFAULT_CHEESE_LEVEL);

        fc_prob_tp_hidden / fc_prob_tp
    } else {
        1.0
    };

    let combo_tps = calculate_combo_tps(&mut map_hit_probs, combo_section_amount);
    let fc_time_tp = *combo_tps.last().expect("no last");
    let (miss_tps, miss_counts) =
        calculate_miss_tps_misscount(&movements, fc_time_tp, miss_section_amount);
    let (cheese_levels, cheese_factor) =
        calculate_cheese_levels_cheese_factors(&movements, fc_prob_tp);
    let cheese_note_count = get_cheese_note_count(&movements, fc_prob_tp);

    AimAttributes {
        fc_prob_throughput: fc_prob_tp,
        hidden_factor,
        combo_throughputs: combo_tps,
        miss_throughputs: miss_tps,
        miss_counts,
        cheese_note_count,
        cheese_levels,
        cheese_factor,
    }
}

#[derive(Default)]
struct AimAttributes {
    fc_prob_throughput: f32,
    hidden_factor: f32,
    combo_throughputs: Vec<f32>,
    miss_throughputs: Vec<f32>,
    miss_counts: Vec<f32>,
    cheese_note_count: f32,
    cheese_levels: Vec<f32>,
    cheese_factor: Vec<f32>,
}

#[inline]
fn get_cheese_note_count(movements: &[Movement], tp: f32) -> f32 {
    movements
        .iter()
        .map(|movement| {
            logistic((movement.index_of_perf / tp - 0.6) * 15.0) * movement.cheesability
        })
        .sum()
}

#[inline]
fn calculate_cheese_levels_cheese_factors(
    movements: &[Movement],
    fc_prob_tp: f32,
) -> (Vec<f32>, Vec<f32>) {
    (0..CHEESE_LEVEL_COUNT)
        .map(|i| {
            let cheese_level = i as f32 / (CHEESE_LEVEL_COUNT - 1) as f32;
            let cheese_factor = calculate_fc_prob_tp(movements, cheese_level) / fc_prob_tp;

            (cheese_level, cheese_factor)
        })
        .unzip()
}

#[inline]
fn calculate_miss_tps_misscount(
    movements: &[Movement],
    fc_time_tp: f32,
    section_amount: usize,
) -> (Vec<f32>, Vec<f32>) {
    let fc_prob = calculate_fc_prob(movements, fc_time_tp, DEFAULT_CHEESE_LEVEL);

    (0..section_amount)
        .map(|i| {
            let miss_tp = fc_time_tp * (1.0 - (i as f32).powf(1.5) * 0.005);
            let miss_probs = get_miss_probs(movements, miss_tp);
            let miss_count = get_miss_count(fc_prob, miss_probs);

            (miss_tp, miss_count)
        })
        .unzip()
}

#[inline]
fn get_miss_probs<'m>(movements: &'m [Movement], tp: f32) -> impl Iterator<Item = f32> + 'm {
    movements.iter().map(move |movement| {
        1.0 - HitProbabilities::calculate_cheese_hit_prob(movement, tp, DEFAULT_CHEESE_LEVEL)
    })
}

fn get_miss_count(p: f32, miss_probs: impl Iterator<Item = f32>) -> f32 {
    let distribution = match PoissonBinomial::new(miss_probs) {
        Some(distribution) => distribution,
        None => return 0.0,
    };

    let cdf_minus_prob = |miss_count| distribution.cdf(miss_count) - p;

    try_expand_find_root_brent(cdf_minus_prob, -100.0, 1000.0, 1e-8, 100, 1.6, 100)
        .expect("no root")
}

#[inline]
fn calculate_combo_tps(hit_probs: &mut HitProbabilities, section_amount: usize) -> Vec<f32> {
    (1..=section_amount)
        .map(|i| calculate_fc_time_tp(hit_probs, i))
        .collect()
}

fn calculate_fc_time_tp(hit_probs: &mut HitProbabilities, section_count: usize) -> f32 {
    let max_fc_time = hit_probs.min_expected_time_for_section_count(TP_MIN, section_count);

    if max_fc_time <= TIME_THRESHOLD_BASE {
        return TP_MIN;
    }

    let min_fc_time = hit_probs.min_expected_time_for_section_count(TP_MAX, section_count);

    if min_fc_time >= TIME_THRESHOLD_BASE {
        return TP_MAX;
    }

    let fc_time_minus_threshold =
        |tp| hit_probs.min_expected_time_for_section_count(tp, section_count) - TIME_THRESHOLD_BASE;

    try_find_root_bisection(
        fc_time_minus_threshold,
        TP_MIN,
        TP_MAX,
        TIME_PRECISION,
        MAX_ITERATIONS,
    )
    .expect("no root")
}

fn calculate_fc_prob(movements: &[Movement], tp: f32, cheese_level: f32) -> f32 {
    let mut fc_prob = 1.0;

    for movement in movements {
        fc_prob *= HitProbabilities::calculate_cheese_hit_prob(movement, tp, cheese_level);
    }

    fc_prob
}

fn calculate_fc_prob_tp(movements: &[Movement], cheese_level: f32) -> f32 {
    let fc_prob_tp_min = calculate_fc_prob(movements, TP_MIN, cheese_level);

    if fc_prob_tp_min >= PROB_TRESHOLD {
        return TP_MIN;
    }

    let fc_prob_tp_max = calculate_fc_prob(movements, TP_MAX, cheese_level);

    if fc_prob_tp_max <= PROB_TRESHOLD {
        return TP_MAX;
    }

    let fc_prob_minus_threshold =
        |tp| calculate_fc_prob(movements, tp, cheese_level) - PROB_TRESHOLD;

    try_find_root_brent(
        fc_prob_minus_threshold,
        TP_MIN,
        TP_MAX,
        PROB_PRECISION,
        MAX_ITERATIONS,
    )
    .expect("no root")
}

fn create_movements(
    hit_objects: &[OsuObject],
    clock_rate: f32,
    strain_history: &[[f32; 4]],
    hidden: bool,
    note_densities: &[f32],
    radius: f32,
) -> Vec<Movement> {
    let mut movements = Movement::extract_movement(&hit_objects[0]);

    for i in 1..hit_objects.len() {
        let obj_neg4 = i.checked_sub(4).and_then(|i| hit_objects.get(i));
        let obj_neg2 = i.checked_sub(2).and_then(|i| hit_objects.get(i));
        let obj_prev = &hit_objects[i - 1];
        let obj_curr = &hit_objects[i];
        let obj_next = hit_objects.get(i + 1);
        let tap_strain = &strain_history[i];

        let note_density = hidden.then(|| note_densities[i]);

        Movement::extract_movement_complete(
            &mut movements,
            obj_neg2,
            obj_prev,
            obj_curr,
            obj_next,
            tap_strain,
            clock_rate,
            hidden,
            note_density,
            obj_neg4,
            radius,
        );
    }

    movements
}

fn calculate_finger_control_diff(hit_objects: &[OsuObject], clock_rate: f32) -> f32 {
    let mut prev_time = hit_objects[0].time / 1000.0;
    let mut curr_strain = 0.0;
    let mut prev_strain_time = 0.0;
    let mut repeat_strain_count = 1;

    let mut strain_history = Vec::with_capacity(hit_objects.len());
    strain_history.push(0.0);

    for h in hit_objects.iter().skip(1) {
        let curr_time = h.time / 1000.0;
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

        if h.is_slider() {
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

    for (i, strain) in strain_history.into_iter().enumerate() {
        diff += strain * k.powi(i as i32);
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

#[derive(Debug)]
struct TapAttributes {
    difficulty: f32,
    stream_note_count: f32,
    mashed_difficulty: f32,
    strain_history: Vec<[f32; 4]>,
}

fn calculate_streamness_mask(hit_objects: &[OsuObject], skill: f32, clock_rate: f32) -> Vec<f32> {
    let mut streamness_mask = Vec::with_capacity(hit_objects.len());
    streamness_mask.push(0.0);

    let stream_time_threshold = skill.powf(-2.7 / 3.2);

    let iter = hit_objects
        .iter()
        .zip(hit_objects.iter().skip(1))
        .map(|(prev, curr)| {
            let t = (curr.time - prev.time) / 1000.0 / clock_rate;

            1.0 - logistic((t / stream_time_threshold - 1.0) * 15.0)
        });

    streamness_mask.extend(iter);

    streamness_mask
}

// Four elements, evenly spaced between 2.3 and -2.8,
// then pointwise applied the exp function on
const DECAY_COEFFS: [f32; 4] = [9.97418, 1.82212, 0.332871, 0.0608101];

const SPACED_BUFF_FACTOR: f32 = 0.1;
const TIMESCALE_FACTOR: [f32; 4] = [1.02, 1.02, 1.05, 1.15];

fn calculate_tap_strain(
    hit_objects: &[OsuObject],
    mash_level: f32,
    clock_rate: f32,
    radius: f32,
) -> (Vec<[f32; 4]>, f32) {
    let mut strain_history = Vec::with_capacity(hit_objects.len());

    strain_history.push([0.0; 4]);
    strain_history.push([0.0; 4]);

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
        strain.pointwise_powf(1.1 / 3.0).scalar_mult(1.5);

        strain_history.push(strain);

        let dist = (hit_objects[i].pos - hit_objects[i - 1].pos).length() / (2.0 * radius);
        let spaced_buf = calculate_spacedness(dist) * SPACED_BUFF_FACTOR;
        let delta_time = ((curr_time - prev_prev_time) / clock_rate).max(0.01);
        let strain_addition =
            (delta_time.powf(-2.7) * 0.265).max((delta_time * delta_time).recip());

        curr_strain.pointwise_add(DECAY_COEFFS.clone().scalar_mult(
            strain_addition
                * calculate_mash_nerf_factor(dist, mash_level).powi(3)
                * (1.0 + spaced_buf).powi(3),
        ));

        prev_prev_time = prev_time;
        prev_time = curr_time;
    }

    let mut strain_sum = 0.0;
    let mut single_strain_history = Vec::with_capacity(hit_objects.len());

    for j in 0..4 {
        let iter = (0..hit_objects.len()).map(|i| strain_history[i][j]);

        single_strain_history.extend(iter);
        single_strain_history.sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap());

        let mut single_strain_result = 0.0;
        let k = 1.0 - 0.04 * DECAY_COEFFS[j].sqrt();

        for (i, strain) in single_strain_history.drain(..).enumerate() {
            single_strain_result += strain * k.powi(i as i32);
        }

        let term = single_strain_result * (1.0 - k) * TIMESCALE_FACTOR[j];
        strain_sum += term * term;
    }

    let diff = (strain_sum / 4.0).sqrt();

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
    let mut note_densities = Vec::with_capacity(hit_objects.len());
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
