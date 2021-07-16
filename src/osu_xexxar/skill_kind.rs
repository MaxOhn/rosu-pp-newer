use super::DifficultyObject;

use rosu_pp::parse::Pos2;
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, LN_2, TAU};

const AIM_BASE_DECAY: f32 = 0.75;
const AIM_HISTORY_LEN: usize = 2;
pub(crate) const AIM_SNAP_STARS_PER_DOUBLE: f32 = 1.125;
pub(crate) const AIM_FLOW_STARS_PER_DOUBLE: f32 = 1.1;
pub(crate) const AIM_COMBINED_STARS_PER_DOUBLE: f32 = 1.15;
const AIM_SNAP_STRAIN_MULTIPLIER: f32 = 6.727;
const AIM_FLOW_STRAIN_MULTIPLIER: f32 = 16.272;
const AIM_HYBRID_STRAIN_MULTIPLIER: f32 = 0.0; // 32.727;
const AIM_SLIDER_STRAIN_MULTIPLIER: f32 = 65.0;
const AIM_TOTAL_STRAIN_MULTIPLIER: f32 = 0.1025;
const AIM_DIST_CONST: f32 = 3.5;
const AIM_BEGIN_DECAY_THRESHOLD: f32 = 500.0;

const TAP_STRAIN_TIME_BUFF_RANGE: f32 = 75.0;
const TAP_BASE_DECAY: f32 = 0.9;
const TAP_HISTORY_LEN: usize = 32;
pub(crate) const TAP_STARS_PER_DOUBLE: f32 = 1.075;
const TAP_SINGLE_MULTIPLIER: f32 = 1.575;
const TAP_STRAIN_MULTIPLIER: f32 = 1.75;
const TAP_RHYTHM_MULTIPLIER: f32 = 1.0;
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
        single_strain: f32,
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
            snap_strains: Vec::with_capacity(128),
            flow_strains: Vec::with_capacity(128),
        }
    }

    #[inline]
    pub(crate) fn new_tap() -> Self {
        Self::Tap {
            curr_strain: 1.0,
            single_strain: 1.0,
            avg_strain_time: 50.0,
            strains: Vec::with_capacity(128),
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

        let computed_decay = self.compute_decay(current.strain_time);

        match self {
            Self::Aim {
                curr_other_strain,
                curr_snap_strain,
                curr_flow_strain,
                snap_strains,
                flow_strains,
            } => {
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
                let hybrid_strain =
                    hybrid_strain_at(prev, curr, next, prev_vec, curr_vec, next_vec);
                let slider_strain = slider_strain_at(prev, curr);

                *curr_snap_strain *= computed_decay;
                *curr_snap_strain +=
                    AIM_TOTAL_STRAIN_MULTIPLIER * snap_strain * AIM_SNAP_STRAIN_MULTIPLIER;

                *curr_flow_strain *= computed_decay;
                *curr_flow_strain +=
                    AIM_TOTAL_STRAIN_MULTIPLIER * flow_strain * AIM_FLOW_STRAIN_MULTIPLIER;

                *curr_other_strain *= computed_decay;
                *curr_other_strain += hybrid_strain * AIM_HYBRID_STRAIN_MULTIPLIER
                    + slider_strain * AIM_SLIDER_STRAIN_MULTIPLIER;

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
                single_strain,
                avg_strain_time,
                strains,
            } => {
                let prev_delta = curr.delta;

                let mut strain_value = 0.25;
                let mut strain_time = ((current.delta).max(50.0) + (prev_delta).max(50.0)) / 2.0;
                let rhythm_complexity = calculate_rhythm_difficulty(previous, avg_strain_time);

                strain_time = strain_time - 25.0;

                strain_value += TAP_STRAIN_TIME_BUFF_RANGE / strain_time;

                *single_strain *= computed_decay;
                *single_strain +=
                    (1.0 + current.snap_probability()) * strain_value * TAP_SINGLE_MULTIPLIER;

                *curr_strain *=
                    computed_decay.powf((current.strain_time / *avg_strain_time).sqrt());
                *curr_strain += strain_value * TAP_STRAIN_MULTIPLIER;

                let strain = single_strain
                    .max((*curr_strain * rhythm_complexity).min(
                        (1.0 - TAP_BASE_DECAY).recip() * strain_value * TAP_STRAIN_MULTIPLIER,
                    ));

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

    #[inline]
    fn base_decay(&self) -> f32 {
        match self {
            SkillKind::Aim { .. } => AIM_BASE_DECAY,
            SkillKind::Tap { .. } => TAP_BASE_DECAY,
        }
    }

    #[inline]
    fn begin_decay_threshold(&self) -> f32 {
        match self {
            SkillKind::Aim { .. } => AIM_BEGIN_DECAY_THRESHOLD,
            SkillKind::Tap { .. } => TAP_BEGIN_DECAY_THRESHOLD,
        }
    }

    #[inline]
    fn compute_decay(&self, ms: f32) -> f32 {
        let begin_decay_threshold = self.begin_decay_threshold();

        if ms < begin_decay_threshold {
            self.base_decay()
        } else {
            (self
                .base_decay()
                .powf(1000.0 / ms.min(begin_decay_threshold)))
            .powf(ms / 1000.0)
        }
    }
}

fn calculate_rhythm_difficulty(
    previous: &VecDeque<DifficultyObject>,
    avg_strain_time: &mut f32,
) -> f32 {
    // let mut island_sizes = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let previous_island_size = -1;
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
                    250.0 * factor
                } else {
                    500.0 * factor
                };
        }

        *avg_strain_time += curr_delta;

        if first_delta_switch {
            if is_ratio_equal(1.0, prev_delta, curr_delta) {
                island_size += 1;
            } else if prev_delta > curr_delta * 1.25 {
                let idx = island_size.min(6);

                island_times[idx] += 100.0 * factor;
                island_sizes[idx] += 1.0;

                if prev_delta > curr_delta * 1.25 {
                    island_size = 0;
                } else {
                    first_delta_switch = false;
                }
            }
        } else if prev_delta > 1.25 * curr_delta {
            first_delta_switch = true;
            island_size = 0;
        }
    }

    let mut rhythm_complexity_sum = 0.0;

    *avg_strain_time /= previous.len() as f32;

    for (&size, time) in island_sizes.iter().zip(island_times.iter()) {
        if size != 0.0 {
            rhythm_complexity_sum += *time / size.sqrt();
        }
    }

    rhythm_complexity_sum += special_transition_count;

    (4.0 + rhythm_complexity_sum * TAP_RHYTHM_MULTIPLIER).sqrt() / 2.0
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

    let avg_vel = (prev_len * prev.flow_probability()
        + curr_len * curr.flow_probability()
        + next_len * next.flow_probability())
        / (prev.flow_probability() + curr.flow_probability() + next.flow_probability()).max(1.0);

    let vel_variance = avg_vel.min(
        (0.5 + 0.5 * prev.flow_probability()) * (avg_vel - prev_len).abs()
            + (0.5 + 0.5 * curr.flow_probability()) * (avg_vel - curr_len).abs()
            + (0.5 + 0.5 * next.flow_probability()) * (avg_vel - next_len).abs(),
    ) / 3.0;

    let angular_variance =
        (curr_len * curr.flow_probability() * FRAC_PI_2.min((next_angle - curr_angle).abs())).min(
            curr_len * curr.flow_probability() * FRAC_PI_2.min((prev_angle - curr_angle).abs()),
        ) / (2.0 * TAU);

    (prev.flow_probability() + curr.flow_probability() + next.flow_probability())
        * (avg_vel + angular_variance.max(vel_variance))
}

#[inline]
fn hybrid_strain_at(
    prev: &DifficultyObject,
    curr: &DifficultyObject,
    next: &DifficultyObject,
    prev_vec: Pos2,
    curr_vec: Pos2,
    next_vec: Pos2,
) -> f32 {
    let prev_len = prev_vec.length();
    let curr_len = curr_vec.length();
    let next_len = next_vec.length();

    let avg_vel = (prev_len + curr_len + next_len) / 3.0;

    let geo_avg_vel = (prev_len * curr_len * next_len).powf(1.0 / 3.0);

    let vel_variance =
        (avg_vel - prev_len).abs() + (avg_vel - curr_len).abs() + (avg_vel - next_len).abs();

    curr.flow_probability() * (geo_avg_vel * vel_variance).sqrt()
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
        * (snap_scaling(prev.jump_dist / 100.0) * (prev.strain_time / (prev.strain_time - 20.0)));
    let curr_vec = curr_vec
        * (snap_scaling(curr.jump_dist / 100.0) * (curr.strain_time / (curr.strain_time - 20.0)));
    let next_vec = next_vec
        * (snap_scaling(next.jump_dist / 100.0) * (next.strain_time / (next.strain_time - 20.0)));

    let prev_len = prev_vec.length();
    let curr_len = curr_vec.length();
    let next_len = next_vec.length();

    let prev_curr_bonus_dist = ((curr_vec + prev_vec).length() - curr_len.max(prev_len)).max(0.0);
    let curr_next_bonus_dist = ((next_vec + curr_vec).length() - next_len.max(curr_len)).max(0.0);

    let avg_vel = (prev_len * prev.snap_probability()
        + curr_len * curr.snap_probability()
        + next_len * next.snap_probability())
        / (prev.snap_probability() + curr.snap_probability() + next.snap_probability()).max(1.0);

    (prev.snap_probability() + curr.snap_probability() + next.snap_probability())
        * (avg_vel
            + (0.25 * avg_vel).min(
                (prev.snap_probability() * curr.snap_probability() * prev_curr_bonus_dist)
                    .max(prev.snap_probability() * curr.snap_probability() * curr_next_bonus_dist)
                    / 2.0,
            ))
}

#[inline]
fn snap_scaling(dist: f32) -> f32 {
    if dist.abs() <= f32::EPSILON {
        0.0
    } else {
        (AIM_DIST_CONST * (dist / AIM_DIST_CONST).ln_1p() / LN_2) / dist
    }
}

#[inline]
fn slider_strain_at(prev: &DifficultyObject, curr: &DifficultyObject) -> f32 {
    (prev.travel_dist / prev.strain_time).max(curr.travel_dist / curr.strain_time)
}
