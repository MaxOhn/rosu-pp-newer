use super::DifficultyObject;

use rosu_pp::parse::Pos2;
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, LN_2};

const TAP_STRAIN_TIME_BUFF_RANGE: f32 = 75.0;
const DECAY_EXCESS_THRESHOLD: f32 = 500.0;

const AIM_BASE_DECAY: f32 = 0.75;
const AIM_HISTORY_LEN: usize = 2;
const AIM_STARS_PER_DOUBLE: f32 = 1.1;
const AIM_SNAP_STRAIN_MULTIPLIER: f32 = 10.0;
const AIM_FLOW_STRAIN_MULTIPLIER: f32 = 15.75;
const AIM_HYBRID_STRAIN_MULTIPLIER: f32 = 8.25;
const AIM_SLIDER_STRAIN_MULTIPLIER: f32 = 75.0;
const AIM_TOTAL_STRAIN_MULTIPLIER: f32 = 0.1675;
const AIM_FITTS_SNAP_CONSTANT: f32 = 3.75;

const TAP_BASE_DECAY: f32 = 0.9;
const TAP_HISTORY_LEN: usize = 16;
const TAP_STARS_PER_DOUBLE: f32 = 1.075;
const TAP_STRAIN_MULTIPLIER: f32 = 2.65;
const TAP_RHYTHM_MULTIPLIER: f32 = 0.75;

#[derive(Copy, Clone, Debug)]
pub(crate) enum SkillKind {
    Aim,
    Tap,
}

impl SkillKind {
    pub(crate) fn strain_value_at(
        self,
        curr_strain: &mut f32,
        current: &DifficultyObject,
        previous: &VecDeque<DifficultyObject>,
    ) -> f32 {
        match self {
            Self::Aim => {
                if current.base.is_spinner() || previous.len() <= 1 {
                    return 0.0;
                }

                let next = current;
                let curr = &previous[0];
                let prev = &previous[1];

                let next_vec = next.dist_vec / next.strain_time;
                let curr_vec = curr.dist_vec / curr.strain_time;
                let prev_vec = prev.dist_vec / prev.strain_time;

                let snap_strain = snap_strain_at(prev, curr);
                let flow_strain = flow_strain_at(prev, curr, next, prev_vec, curr_vec);
                let hybrid_strain =
                    hybrid_strain_at(prev, curr, next, prev_vec, curr_vec, next_vec);
                let slider_strain = slider_strain_at(prev);

                *curr_strain *= self.compute_decay(current.strain_time);

                *curr_strain += snap_strain * AIM_SNAP_STRAIN_MULTIPLIER;
                *curr_strain += flow_strain * AIM_FLOW_STRAIN_MULTIPLIER;
                *curr_strain += hybrid_strain * AIM_HYBRID_STRAIN_MULTIPLIER;
                *curr_strain += slider_strain * AIM_SLIDER_STRAIN_MULTIPLIER;

                *curr_strain * AIM_TOTAL_STRAIN_MULTIPLIER
            }
            Self::Tap => {
                if current.base.is_spinner() {
                    return 0.0;
                }

                let (_, prev_strain_time) = match current.prev {
                    Some(tuple) => tuple,
                    None => return 0.0,
                };

                let mut strain_value = 0.25;
                let avg_delta_time = (current.strain_time + prev_strain_time) / 2.0;
                let rhythm_complexity = calculate_rhythm_difficulty(previous);

                let div = TAP_STRAIN_TIME_BUFF_RANGE / avg_delta_time;

                if div > 1.0 {
                    strain_value += div * div;
                } else {
                    strain_value += div;
                }

                *curr_strain *= self.compute_decay(current.strain_time);
                *curr_strain += strain_value * TAP_STRAIN_MULTIPLIER;

                *curr_strain * (previous.len() / TAP_HISTORY_LEN) as f32 * rhythm_complexity
            }
        }
    }

    #[inline]
    pub(crate) fn history_len(&self) -> usize {
        match self {
            Self::Aim => AIM_HISTORY_LEN,
            Self::Tap => TAP_HISTORY_LEN,
        }
    }

    #[inline]
    fn base_decay(&self) -> f32 {
        match self {
            SkillKind::Aim => AIM_BASE_DECAY,
            SkillKind::Tap => TAP_BASE_DECAY,
        }
    }

    #[inline]
    pub(crate) fn difficulty_exponent(&self) -> f32 {
        1.0 / self.stars_per_double().log2()
    }

    #[inline]
    fn stars_per_double(&self) -> f32 {
        match self {
            SkillKind::Aim => AIM_STARS_PER_DOUBLE,
            SkillKind::Tap => TAP_STARS_PER_DOUBLE,
        }
    }

    #[inline]
    fn compute_decay(&self, ms: f32) -> f32 {
        if ms < DECAY_EXCESS_THRESHOLD {
            self.base_decay()
        } else {
            (self
                .base_decay()
                .powf(1000.0 / ms.min(DECAY_EXCESS_THRESHOLD)))
            .powf(ms / 1000.0)
        }
    }
}

fn calculate_rhythm_difficulty(previous: &VecDeque<DifficultyObject>) -> f32 {
    let mut island_sizes = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut island_times = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut island_size = 0;
    let mut special_transition_count = 0.0;

    let mut first_delta_switch = false;

    for i in 1..previous.len() {
        let prev_delta = previous[i - 1].strain_time;
        let curr_delta = previous[i].strain_time;

        let factor = (i as f32 / TAP_HISTORY_LEN as f32) / (prev_delta * curr_delta).sqrt();

        if is_ratio_equal(1.5, prev_delta, curr_delta)
            || is_ratio_equal(1.5, curr_delta, prev_delta)
        {
            special_transition_count +=
                if previous[i - 1].base.is_slider() || previous[i].base.is_slider() {
                    50.0 * factor
                } else {
                    250.0 * factor
                };
        }

        if first_delta_switch {
            if is_ratio_equal(1.0, prev_delta, curr_delta) {
                island_size += 1;
            } else {
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
) -> f32 {
    let prev_diff_vec = prev_vec - curr_vec;
    let angle = (curr.angle.unwrap_or(0.0) + next.angle.unwrap_or(0.0)) / 2.0;

    let angle_adjustment = if angle < FRAC_PI_4 {
        0.0
    } else if angle > FRAC_PI_2 {
        (prev_diff_vec.length() - curr_vec.length().max(prev_vec.length()))
            .min(100.0 / curr.strain_time)
            .max(0.0)
    } else {
        let base = (2.0 * (angle - FRAC_PI_4)).sin();

        ((curr.jump_dist - 75.0).max(0.0) / 50.0).min(1.0)
            * base
            * base
            * (prev_diff_vec.length() - curr_vec.length().max(prev_vec.length()) / 2.0)
                .min(100.0 / curr.strain_time)
                .max(0.0)
    };

    let strain = prev_vec.length() * prev.flow_probability()
        + curr_vec.length() * curr.flow_probability()
        + curr_vec
            .length()
            .min(prev_vec.length())
            .min((curr_vec.length() - prev_vec.length()).abs())
            * curr.flow_probability()
            * prev.flow_probability()
        + angle_adjustment * curr.flow_probability() * prev.flow_probability();

    let factor = (curr.strain_time / (curr.strain_time - 10.0))
        .min(prev.strain_time / (prev.strain_time - 10.0));

    strain * factor
}

#[inline]
fn snap_strain_at(prev: &DifficultyObject, curr: &DifficultyObject) -> f32 {
    let curr_vec = (curr.dist_vec * snap_scaling(curr.jump_dist / 100.0)) / curr.strain_time;
    let prev_vec = (prev.dist_vec * snap_scaling(prev.jump_dist / 100.0)) / prev.strain_time;

    let prev_diff_vec = prev_vec + curr_vec;
    let angle_dist = (prev_diff_vec.length() - curr_vec.length().max(prev_vec.length())).max(0.0);

    let mut angle_adjustment = 0.0;
    let curr_dist =
        curr_vec.length() * curr.snap_probability() + prev_vec.length() * prev.snap_probability();
    let angle = curr.angle.unwrap_or(0.0).abs();

    if angle < FRAC_PI_3 {
        let base = (FRAC_PI_2 - angle * 1.5).sin();

        angle_adjustment -= 0.2 * (curr_dist - angle_dist).abs() * base * base;
    } else {
        let base = (angle - FRAC_PI_4).sin();

        angle_adjustment += angle_dist * (1.0 + 0.5 * base * base);
    }

    let strain = curr_dist + angle_adjustment * curr.snap_probability() * prev.snap_probability();

    let factor = (curr.strain_time / (curr.strain_time - 20.0))
        .min(prev.strain_time / (prev.strain_time - 20.0));

    strain * factor
}

#[inline]
fn snap_scaling(dist: f32) -> f32 {
    if dist != 0.0 {
        AIM_FITTS_SNAP_CONSTANT * ((dist / AIM_FITTS_SNAP_CONSTANT).ln_1p() / LN_2) / dist
    } else {
        0.0
    }
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
    let flow_to_snap_vec = prev_vec - curr_vec;
    let snap_to_flow_vec = curr_vec + next_vec;

    let flow_to_snap_strain =
        flow_to_snap_vec.length() * curr.snap_probability() * prev.flow_probability();
    let snap_to_flow_strain =
        snap_to_flow_vec.length() * curr.snap_probability() * next.flow_probability();

    let flow_to_snap_value = flow_to_snap_strain * (curr_vec.length() * prev_vec.length()).sqrt();
    let snap_to_flow_value = snap_to_flow_strain * (curr_vec.length() * next_vec.length()).sqrt();

    flow_to_snap_value.max(snap_to_flow_value).sqrt()
}

#[inline]
fn slider_strain_at(prev: &DifficultyObject) -> f32 {
    prev.travel_dist / prev.strain_time
}
