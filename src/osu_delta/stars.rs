use crate::osu_delta::math_util::try_expand_find_root_brent;

use super::math_util::{
    calculate_hit_prob, logistic, try_find_root_bisection, try_find_root_brent, PoissonBinomial,
};
use super::{
    difficulty_range_ar, difficulty_range_od, ArrayVec, DifficultyAttributes, Movement, OsuObject,
    SliderState,
};

use rosu_pp::{Beatmap, Mods};
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
) -> AimAttributes {
    let movements = create_movements(hit_objects, clock_rate, strain_history, false, None, radius);

    let movement_hidden = create_movements(
        hit_objects,
        clock_rate,
        strain_history,
        true,
        Some(note_densities),
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
    let fc_prob_tp_hidden = calculate_fc_prob_tp(&movement_hidden, DEFAULT_CHEESE_LEVEL);

    let hidden_factor = fc_prob_tp_hidden / fc_prob_tp;

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

fn get_cheese_note_count(movements: &[Movement], tp: f32) -> f32 {
    let mut count = 0.0;

    for movement in movements {
        count += logistic((movement.index_of_perf / tp - 0.6) * 15.0) * movement.cheesability;
    }

    count
}

fn calculate_cheese_levels_cheese_factors(
    movements: &[Movement],
    fc_prob_tp: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut cheese_levels = vec![0.0; CHEESE_LEVEL_COUNT];
    let mut cheese_factors = vec![0.0; CHEESE_LEVEL_COUNT];

    for i in 0..CHEESE_LEVEL_COUNT {
        let cheese_level = i as f32 / (CHEESE_LEVEL_COUNT - 1) as f32;
        cheese_levels[i] = cheese_level;
        cheese_factors[i] = calculate_fc_prob_tp(movements, cheese_level) / fc_prob_tp;
    }

    (cheese_levels, cheese_factors)
}

fn calculate_miss_tps_misscount(
    movements: &[Movement],
    fc_time_tp: f32,
    section_amount: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut miss_tps = vec![0.0; section_amount];
    let mut miss_counts = vec![0.0; section_amount];
    let fc_prob = calculate_fc_prob(movements, fc_time_tp, DEFAULT_CHEESE_LEVEL);

    for i in 0..section_amount {
        let miss_tp = fc_time_tp * (1.0 - (i as f32).powf(1.5) * 0.005);
        let miss_probs = get_miss_probs(movements, miss_tp);
        miss_tps[i] = miss_tp;
        miss_counts[i] = get_miss_count(fc_prob, &miss_probs);
    }

    (miss_tps, miss_counts)
}

fn get_miss_probs(movements: &[Movement], tp: f32) -> Vec<f32> {
    let mut miss_probs = vec![0.0; movements.len()];

    for i in 0..movements.len() {
        let movement = &movements[i];
        miss_probs[i] =
            1.0 - HitProbabilities::calculate_cheese_hit_prob(&movement, tp, DEFAULT_CHEESE_LEVEL);
    }

    miss_probs
}

fn get_miss_count(p: f32, miss_probs: &[f32]) -> f32 {
    if miss_probs.iter().sum::<f32>().abs() < f32::EPSILON {
        return 0.0;
    }

    let distribution = PoissonBinomial::new(miss_probs);
    let cdf_minus_prob = |miss_count| distribution.cdf(miss_count) - p;

    try_expand_find_root_brent(cdf_minus_prob, -100.0, 1000.0, 1e-8, 100, 1.6, 100)
        .expect("no root")
}

fn calculate_combo_tps(hit_probs: &mut HitProbabilities, section_amount: usize) -> Vec<f32> {
    let mut combo_tps = vec![0.0; section_amount];

    for i in 1..=section_amount {
        combo_tps[i - 1] = calculate_fc_time_tp(hit_probs, i);
    }

    combo_tps
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

fn calculate_fc_prob(movements: &[Movement], tp: f32, cheese_level: f32) -> f32 {
    let mut fc_prob = 1.0;

    for movement in movements {
        fc_prob *= HitProbabilities::calculate_cheese_hit_prob(movement, tp, cheese_level);
    }

    fc_prob
}

#[derive(Copy, Clone)]
struct SkillData {
    expected_time: f32,
    fc_prob: f32,
}

impl Default for SkillData {
    fn default() -> Self {
        Self {
            expected_time: 0.0,
            fc_prob: 1.0,
        }
    }
}

use std::collections::HashMap;

struct MapSectionCache<'m> {
    cache: HashMap<F32, SkillData>,
    cheese_level: f32,
    start_t: f32,
    end_t: f32,
    movements: &'m [Movement],
}

#[derive(Copy, Clone, PartialEq)]
struct F32(f32);

impl Eq for F32 {}

impl Hash for F32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.0 as u32);
        state.finish();
    }
}

impl<'m> MapSectionCache<'m> {
    fn new(movements: &'m [Movement], cheese_level: f32, start_t: f32, end_t: f32) -> Self {
        Self {
            movements,
            start_t,
            end_t,
            cheese_level,
            cache: HashMap::new(),
        }
    }

    fn evaluate(&mut self, tp: f32) -> SkillData {
        let mut result = SkillData::default();

        if self.movements.is_empty() {
            return result;
        }

        let key = F32(tp);

        if let Some(result) = self.cache.get(&key) {
            return *result;
        }

        for movement in self.movements {
            let mut hit_prob =
                HitProbabilities::calculate_cheese_hit_prob(movement, tp, self.cheese_level)
                    + 1e-10;
            hit_prob = 1.0 - ((1.0 - hit_prob + 0.25).sqrt() - 0.5);

            result.expected_time = (result.expected_time + movement.raw_movement_time) / hit_prob;
            result.fc_prob *= hit_prob;
        }

        self.cache.insert(key, result);

        result
    }
}

struct HitProbabilities<'m> {
    sections: Vec<MapSectionCache<'m>>,
}

impl<'m> HitProbabilities<'m> {
    fn new(movements: &'m [Movement], cheese_level: f32, difficulty_count: usize) -> Self {
        let mut sections = Vec::with_capacity(difficulty_count);

        for i in 0..difficulty_count {
            let start = movements.len() * i / difficulty_count;
            let end = movements.len() * (i + 1) / difficulty_count - 1;
            let start_t = movements[start].time;
            let end_t = movements[end].time;

            sections.push(MapSectionCache::new(
                &movements[start..end + 1],
                cheese_level,
                start_t,
                end_t,
            ));
        }

        Self { sections }
    }

    fn min_expected_time_for_section_count(&mut self, tp: f32, section_count: usize) -> f32 {
        let mut fc_time = f32::INFINITY;
        let section_data: Vec<_> = self.sections.iter_mut().map(|s| s.evaluate(tp)).collect();

        for i in 0..=self.sections.len() - section_count {
            fc_time = fc_time.min(
                expected_fc_time(&section_data, i, section_count) - self.length(i, section_count),
            );
        }

        fc_time
    }

    #[inline]
    fn length(&self, start: usize, section_count: usize) -> f32 {
        self.sections[start + section_count - 1].end_t - self.sections[start].start_t
    }

    fn calculate_cheese_hit_prob(movement: &Movement, tp: f32, cheese_level: f32) -> f32 {
        let mut per_movement_cheese_level = cheese_level;

        if movement.ends_on_slider {
            per_movement_cheese_level = 0.5 * cheese_level + 0.5;
        }

        let cheese_mt =
            movement.movement_time * (1.0 + per_movement_cheese_level * movement.cheesable_ratio);

        calculate_hit_prob(movement.dist, cheese_mt, tp)
    }
}

fn expected_fc_time(section_data: &[SkillData], start: usize, count: usize) -> f32 {
    let mut fc_time = 15.0;

    for i in start..start + count {
        fc_time /= section_data[i].fc_prob;
        fc_time += section_data[i].expected_time;
    }

    fc_time
}

fn create_movements(
    hit_objects: &[OsuObject],
    clock_rate: f32,
    strain_history: &[[f32; 4]],
    hidden: bool,
    note_densities: Option<&[f32]>,
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

        let density = note_densities.filter(|_| hidden).map(|d| d[i]);

        let new_movements = Movement::extract_movement_complete(
            obj_neg2,
            obj_prev,
            obj_curr,
            obj_next,
            Some(tap_strain),
            clock_rate,
            hidden,
            density,
            obj_neg4,
            radius,
        );

        movements.extend(new_movements);
    }

    movements
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

#[derive(Debug)]
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

// Four elements, evenly spaced between 2.3 and -2.8,
// then pointwise applied the exp function on
const DECAY_COEFFS: [f32; 4] = [9.97418, 1.82212, 0.332871, 0.0608101];

use std::cmp::Ordering;
use std::hash::Hash;

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

    let mut strain_result = [0.0; 4];

    for j in 0..1 {
        let mut single_strain_history = vec![0.0; hit_objects.len()];

        for i in 0..hit_objects.len() {
            single_strain_history[i] = strain_history[i][j];
        }

        single_strain_history.sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap());

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
