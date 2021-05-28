use super::OsuObject;

use rosu_pp::parse::Pos2;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

#[derive(Clone, Debug)]
pub(crate) struct DifficultyObject<'h> {
    pub(crate) base: &'h OsuObject,
    pub(crate) prev: Option<(f32, f32)>, // (jump_dist, strain_time)

    pub(crate) dist_vec: Pos2,
    pub(crate) jump_dist: f32,
    pub(crate) travel_dist: f32,
    pub(crate) angle: Option<f32>,

    pub(crate) delta: f32,
    pub(crate) strain_time: f32,

    flow_probability: f32,
}

impl<'h> DifficultyObject<'h> {
    pub(crate) fn new(
        base: &'h OsuObject,
        prev: &OsuObject,
        prev_vals: Option<(f32, f32)>, // (jump_dist, strain_time)
        prev_prev: Option<&'h OsuObject>,
        clock_rate: f32,
        scaling_factor: f32,
    ) -> Self {
        let delta = (base.time - prev.time) / clock_rate;
        let strain_time = delta.max(50.0);

        let pos = base.pos;
        let travel_dist = prev.travel_dist;
        let prev_cursor_pos = prev.end_pos;

        let dist_vec = if base.is_spinner() {
            Pos2::zero()
        } else {
            (pos - prev_cursor_pos) * scaling_factor
        };

        let jump_dist = dist_vec.length();

        let angle = prev_prev.map(|prev_prev| {
            let prev_prev_cursor_pos = prev_prev.end_pos;

            let v1 = prev_prev_cursor_pos - prev.pos;
            let v2 = pos - prev_cursor_pos;

            let dot = v1.dot(v2);
            let det = v1.x * v2.y - v1.y * v2.x;

            det.atan2(dot).abs()
        });

        let prob_angle = angle.unwrap_or(0.0).clamp(FRAC_PI_6, FRAC_PI_2);
        let angle_offset = 10.0 * (1.5 * (FRAC_PI_2 - prob_angle)).sin();

        let dist_offset = jump_dist.powf(1.7) / 325.0;
        let flow_probability = (1.0 + (delta - 126.0 + dist_offset + angle_offset).exp()).recip();

        Self {
            base,
            prev: prev_vals,

            dist_vec,
            jump_dist,
            travel_dist,
            angle,

            delta,
            strain_time,

            flow_probability,
        }
    }

    #[inline]
    pub(crate) fn flow_probability(&self) -> f32 {
        self.flow_probability
    }

    #[inline]
    pub(crate) fn snap_probability(&self) -> f32 {
        1.0 - self.flow_probability
    }
}
