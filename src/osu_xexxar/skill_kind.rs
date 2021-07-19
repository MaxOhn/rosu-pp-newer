use super::DifficultyObject;

use rosu_pp::parse::Pos2;
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, LN_2};

const OBJ_COUNT_ESTIMATE: usize = 256;

const AIM_BASE_DECAY: f32 = 0.75;
const AIM_HISTORY_LEN: usize = 2;
const AIM_SNAP_STRAIN_MULTIPLIER: f32 = 7.3727;
const AIM_FLOW_STRAIN_MULTIPLIER: f32 = 16.272;
const AIM_SLIDER_STRAIN_MULTIPLIER: f32 = 65.0;
const AIM_TOTAL_STRAIN_MULTIPLIER: f32 = 0.1025;
const AIM_SNAP_DIST_CONST: f32 = 5.0;
const AIM_BEGIN_DECAY_THRESHOLD: f32 = 500.0;

const TAP_STRAIN_TIME_BUFF_RANGE: f32 = 75.0;
const TAP_BASE_DECAY: f32 = 0.9;
const TAP_HISTORY_LEN: usize = 32;
const TAP_STRAIN_MULTIPLIER: f32 = 1.725;
const TAP_RHYTHM_MULTIPLIER: f32 = 0.625;
const TAP_BEGIN_DECAY_THRESHOLD: f32 = 500.0;

#[derive(Clone, Debug)]
pub(crate) enum SkillKind {
    Aim {
        curr_other_strain: f32,
        curr_snap_strain: f32,
        curr_flow_strain: f32,

        snap_strains: Vec<f32>,
        flow_strains: Vec<f32>,
    },
    Tap {
        curr_strain: f32,
        avg_strain_time: f32,

        strains: Vec<f32>,
    },
}

impl SkillKind {
    #[inline]
    pub(crate) fn new_aim() -> Self {
        Self::Aim {
            curr_other_strain: 1.0,
            curr_snap_strain: 1.0,
            curr_flow_strain: 1.0,
            snap_strains: Vec::with_capacity(OBJ_COUNT_ESTIMATE),
            flow_strains: Vec::with_capacity(OBJ_COUNT_ESTIMATE),
        }
    }

    #[inline]
    pub(crate) fn new_tap() -> Self {
        Self::Tap {
            curr_strain: 1.0,
            avg_strain_time: 50.0,
            strains: Vec::with_capacity(OBJ_COUNT_ESTIMATE),
        }
    }

    pub(crate) fn process(
        &mut self,
        current: &DifficultyObject,
        previous: &VecDeque<DifficultyObject>,
    ) {
        let curr = match previous.get(0) {
            Some(curr) => curr,
            None => return,
        };

        match self {
            Self::Aim {
                curr_other_strain,
                curr_snap_strain,
                curr_flow_strain,
                snap_strains,
                flow_strains,
            } => {
                let computed_decay = compute_decay(
                    AIM_BASE_DECAY,
                    current.strain_time,
                    AIM_BEGIN_DECAY_THRESHOLD,
                );

                let next = current;

                let prev = match previous.get(1) {
                    Some(prev) => prev,
                    None => return,
                };

                let next_vec = next.dist_vec / next.strain_time;
                let curr_vec = curr.dist_vec / curr.strain_time;
                let prev_vec = prev.dist_vec / prev.strain_time;

                let snap_strain = snap_strain_at(prev, curr, next, prev_vec, curr_vec, next_vec);
                let flow_strain = flow_strain_at(prev, curr, next, prev_vec, curr_vec, next_vec);
                let slider_strain = slider_strain_at(prev, curr);

                *curr_snap_strain *= computed_decay;
                *curr_snap_strain += snap_strain * AIM_SNAP_STRAIN_MULTIPLIER;

                *curr_flow_strain *= computed_decay;
                *curr_flow_strain += flow_strain * AIM_FLOW_STRAIN_MULTIPLIER;

                *curr_other_strain *= computed_decay;
                *curr_other_strain += slider_strain * AIM_SLIDER_STRAIN_MULTIPLIER;

                let total_strain = AIM_TOTAL_STRAIN_MULTIPLIER
                    * (*curr_snap_strain + *curr_flow_strain + *curr_other_strain);

                if *curr_snap_strain > *curr_flow_strain {
                    snap_strains.push(total_strain);
                } else {
                    flow_strains.push(total_strain);
                }
            }
            Self::Tap {
                curr_strain,
                avg_strain_time,
                strains,
            } => {
                let computed_decay = compute_decay(
                    TAP_BASE_DECAY,
                    current.strain_time,
                    TAP_BEGIN_DECAY_THRESHOLD,
                );

                let prev_delta = curr.delta;

                let mut strain_value = 0.25;
                let mut strain_time = (current.delta + prev_delta) / 2.0;

                if strain_time < 50.0 {
                    strain_time = (strain_time + 150.0) / 4.0;
                }

                let rhythm_complexity = calculate_rhythm_difficulty(previous, avg_strain_time);

                strain_time = strain_time - 25.0;

                strain_value += TAP_STRAIN_TIME_BUFF_RANGE / strain_time;

                *curr_strain *=
                    computed_decay.powf((current.strain_time / *avg_strain_time).max(1.0));

                *curr_strain +=
                    (1.0 + 0.5 * curr.snap_probability()) * strain_value * TAP_STRAIN_MULTIPLIER;

                let strain = *curr_strain * rhythm_complexity;

                strains.push(strain);
            }
        }
    }

    #[inline]
    pub(crate) fn history_len(&self) -> usize {
        match self {
            Self::Aim { .. } => AIM_HISTORY_LEN,
            Self::Tap { .. } => TAP_HISTORY_LEN,
        }
    }
}

fn calculate_rhythm_difficulty(
    previous: &VecDeque<DifficultyObject>,
    avg_strain_time: &mut f32,
) -> f32 {
    let mut previous_island_size = usize::MAX;
    let mut island_times = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut island_size = 0;
    let mut special_transition_count = 0.0;

    let mut first_delta_switch = false;

    for i in 1..previous.len() {
        let prev_delta = previous[i - 1].strain_time;
        let curr_delta = previous[i].strain_time;
        let geo_avg_delta = (prev_delta * curr_delta).sqrt();

        let factor = (i as f32 / TAP_HISTORY_LEN as f32) / geo_avg_delta;

        if is_ratio_equal(1.5, prev_delta, curr_delta)
            || is_ratio_equal(1.5, curr_delta, prev_delta)
        {
            special_transition_count +=
                if previous[i - 1].base.is_slider() || previous[i].base.is_slider() {
                    100.0 * factor
                } else {
                    200.0 * factor
                };
        }

        *avg_strain_time += curr_delta;

        if first_delta_switch {
            if is_ratio_equal(1.0, prev_delta, curr_delta) {
                island_size += 1;
            } else if prev_delta > curr_delta * 1.25 {
                if island_size > 6 {
                    if previous_island_size == 6 {
                        island_times[6] += 100.0 * factor;
                    } else {
                        island_times[6] += 200.0 * factor;
                    }

                    previous_island_size = 6;
                } else {
                    if previous_island_size == island_size {
                        island_times[island_size] += 100.0 * factor;
                    } else {
                        island_times[island_size] += 200.0 * factor;
                    }

                    previous_island_size = island_size;
                }

                if prev_delta > curr_delta * 1.25 {
                    first_delta_switch = false;
                }

                island_size = 0;
            }
        } else if prev_delta > 1.25 * curr_delta {
            first_delta_switch = true;
            island_size = 0;
        }
    }

    let rhythm_complexity_sum = island_times.iter().sum::<f32>() + special_transition_count;
    *avg_strain_time /= previous.len() as f32;

    ((4.0 + rhythm_complexity_sum * TAP_RHYTHM_MULTIPLIER).sqrt() / 2.0).min(1.5)
}

#[inline]
fn is_ratio_equal(ratio: f32, a: f32, b: f32) -> bool {
    a + 15.0 > ratio * b && a - 15.0 < ratio * b
}

#[inline]
fn flow_strain_at(
    prev: &DifficultyObject,
    curr: &DifficultyObject,
    next: &DifficultyObject,
    prev_vec: Pos2,
    curr_vec: Pos2,
    next_vec: Pos2,
) -> f32 {
    let prev_vec = prev_vec * (prev.strain_time / (prev.strain_time - 10.0));
    let curr_vec = curr_vec * (curr.strain_time / (curr.strain_time - 10.0));
    let next_vec = next_vec * (next.strain_time / (next.strain_time - 10.0));

    let prev_len = prev_vec.length();
    let curr_len = curr_vec.length();
    let next_len = next_vec.length();

    let prev_angle = prev.angle.unwrap_or(0.0);
    let curr_angle = curr.angle.unwrap_or(0.0);
    let next_angle = next.angle.unwrap_or(0.0);

    let min_dist = prev_len.min(curr_len).min(next_len);

    let avg_vel = (prev_len * prev.flow_probability()
        + curr_len * curr.flow_probability()
        + next_len * next.flow_probability())
        / (prev.flow_probability() + curr.flow_probability() + next.flow_probability()).max(1.0);

    let vel_variance = 0.65 * (min_dist * curr.flow_probability() * (avg_vel - curr_len).max(0.0));

    let angular_variance = 0.25
        * (min_dist * curr.flow_probability() * FRAC_PI_2.min((next_angle - curr_angle).max(0.0)))
            .min(
                min_dist
                    * curr.flow_probability()
                    * FRAC_PI_2.min((prev_angle - curr_angle).max(0.0)),
            );

    (prev.flow_probability() + curr.flow_probability() + next.flow_probability())
        * (avg_vel + angular_variance.max(vel_variance))
}

#[inline]
fn snap_strain_at(
    prev: &DifficultyObject,
    curr: &DifficultyObject,
    next: &DifficultyObject,
    prev_vec: Pos2,
    curr_vec: Pos2,
    next_vec: Pos2,
) -> f32 {
    let prev_vec = prev_vec
        * (snap_scaling(prev.jump_dist / 100.0) * (prev.strain_time / (prev.strain_time - 15.0)));
    let curr_vec = curr_vec
        * (snap_scaling(curr.jump_dist / 100.0) * (curr.strain_time / (curr.strain_time - 15.0)));
    let next_vec = next_vec
        * (snap_scaling(next.jump_dist / 100.0) * (next.strain_time / (next.strain_time - 15.0)));

    let prev_len = prev_vec.length();
    let curr_len = curr_vec.length();
    let next_len = next_vec.length();

    let prev_bonus_dist = curr.snap_probability()
        * ((curr_vec + prev_vec).length() - curr_len.max(prev_len) / 1.5).max(0.0);

    let avg_vel = (prev_len * prev.snap_probability()
        + curr_len * curr.snap_probability()
        + next_len * next.snap_probability())
        / (prev.snap_probability() + curr.snap_probability() + next.snap_probability()).max(1.0);

    (prev.snap_probability() + curr.snap_probability() + next.snap_probability())
        * (avg_vel + (avg_vel / 2.0).min(prev_bonus_dist))
}

#[inline]
fn snap_scaling(dist: f32) -> f32 {
    if dist.abs() <= f32::EPSILON {
        0.0
    } else {
        (AIM_SNAP_DIST_CONST * (dist / AIM_SNAP_DIST_CONST).ln_1p() / LN_2) / dist
    }
}

#[inline]
fn slider_strain_at(prev: &DifficultyObject, curr: &DifficultyObject) -> f32 {
    (prev.travel_dist / prev.strain_time).max(curr.travel_dist / curr.strain_time)
}

#[inline]
fn compute_decay(base_decay: f32, ms: f32, begin_decay_threshold: f32) -> f32 {
    if ms < begin_decay_threshold {
        base_decay
    } else {
        (base_decay.powf(1000.0 / ms.min(begin_decay_threshold))).powf(ms / 1000.0)
    }
}
