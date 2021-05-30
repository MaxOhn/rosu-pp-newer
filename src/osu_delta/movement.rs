use super::{math_util::logistic, OsuObject};

use rosu_pp::parse::Pos2;

const T_RATIO_THRESHOLD: f32 = 1.4;
const CORRECTION_NEG2_STILL: f32 = 0.0;

#[derive(Default)]
pub struct Movement {
    raw_movement_time: f32,
    dist: f32,
    movement_time: f32,
    index_of_perf: f32,
    cheesability: f32,
    cheesable_ratio: f32,
    time: f32,
    ends_on_slider: bool,
}

impl Movement {
    fn empty(time: f32) -> Self {
        Self {
            dist: 0.0,
            movement_time: 1.0,
            cheesable_ratio: 0.0,
            cheesability: 0.0,
            raw_movement_time: 0.0,
            index_of_perf: 0.0,
            time,
            ..Default::default()
        }
    }

    fn extract_movement(h: &OsuObject) -> Vec<Self> {
        let init_time = h.time / 1000.0;
        let mut movement_with_nested = vec![Self::empty(init_time)];

        if h.nested_object_count > 0 {
            let extra_nested_count = h.nested_object_count - 1;

            for i in 0..extra_nested_count {
                movement_with_nested.push(Self::empty(init_time));
            }
        }

        movement_with_nested
    }

    fn extract_movement_complete(
        obj_neg2: &OsuObject,
        obj_prev: &OsuObject,
        obj_curr: &OsuObject,
        obj_next: &OsuObject,
        tap_strain: &[f32; 4],
        clock_rate: f32,
        hidden: bool,
        note_density: Option<f32>,
        obj_neg4: Option<&OsuObject>,
        radius: f32,
    ) -> Vec<Self> {
        let note_density = note_density.unwrap_or(0.0);

        let mut movement = Self::default();

        let t_prev_curr = (obj_curr.time - obj_prev.time) / clock_rate / 1000.0;
        movement.raw_movement_time = t_prev_curr;
        movement.time = obj_curr.time / 1000.0;

        if obj_curr.is_spinner() || obj_prev.is_spinner() {
            movement.movement_time = 1.0;

            return vec![movement];
        }

        let obj_neg2 = (!obj_neg2.is_spinner()).then(|| obj_neg2);
        let obj_next = (!obj_next.is_spinner()).then(|| obj_next);

        if obj_curr.is_slider() {
            movement.ends_on_slider = true;
        }

        let pos_prev = obj_prev.pos;
        let pos_curr = obj_curr.pos;
        let s_prev_curr = (pos_curr - pos_prev) / (2.0 * radius);

        let d_prev_curr = s_prev_curr.length();
        let ip_prev_curr = calculate_ip(d_prev_curr, t_prev_curr);

        movement.index_of_perf = ip_prev_curr;

        let mut pos_neg2 = Pos2::zero();
        let mut pos_next = Pos2::zero();
        let mut s_neg2_prev = Pos2::zero();
        let mut s_curr_next = Pos2::zero();

        let mut d_neg2_prev = 0.0;
        let mut d_neg2_curr = 0.0;
        let mut d_curr_next = 0.0;
        let mut t_neg2_prev = 0.0;
        let mut t_curr_next = 0.0;

        let mut flowiness_neg2_prev_curr = 0.0;
        let mut flowiness_prev_curr_next = 0.0;
        let mut obj_prev_temp_in_mid = false;
        let mut obj_curr_temp_in_mid = false;

        let mut d_neg4_curr = 0.0;

        if let Some(obj_neg4) = obj_neg4 {
            let pos_neg4 = obj_neg4.pos;
            d_neg4_curr = ((pos_curr - pos_neg4) / (2.0 * radius)).length();
        }

        if let Some(obj_neg2) = obj_neg2 {
            pos_neg2 = obj_neg2.pos;
            s_neg2_prev = (pos_prev - pos_neg2) / (2.0 * radius);
            d_neg2_prev = s_neg2_prev.length();
            t_neg2_prev = (obj_prev.time - obj_neg2.time) / clock_rate / 1000.0;
            d_neg2_curr = ((pos_curr - pos_neg2) / (2.0 * radius)).length();
        }

        if let Some(obj_next) = obj_next {
            pos_next = obj_next.pos;
            s_curr_next = (pos_next - pos_curr) / (2.0 * radius);
            d_curr_next = s_curr_next.length();
            t_curr_next = (obj_next.time - obj_curr.time) / clock_rate / 1000.0;
        }

        let mut correction_neg2 = 0.0;

        if let Some(obj_neg2) = obj_neg2.filter(|_| d_prev_curr != 0.0) {
            let t_ratio_neg2 = t_prev_curr / t_neg2_prev;
            let cos_neg2_prev_curr = (s_neg2_prev.dot(s_prev_curr) / -d_neg2_prev / d_prev_curr)
                .max(-1.0)
                .min(1.0);

            if t_ratio_neg2 > T_RATIO_THRESHOLD {
                if d_neg2_prev.abs() < f32::EPSILON {
                    correction_neg2 = CORRECTION_NEG2_STILL;
                } else {
                    let correction_neg2_moving = interpolate(cos_neg2_prev_curr);
                    let movingness = logistic(d_neg2_prev * 6.0 - 5.0) - logistic(-5.0);

                    correction_neg2 = (movingness * correction_neg2_moving
                        + (1.0 - movingness) * CORRECTION_NEG2_STILL)
                        * 1.5;
                }
            } else if t_ratio_neg2 < T_RATIO_THRESHOLD.recip() {
                if d_neg2_prev.abs() < f32::EPSILON {
                    correction_neg2 = 0.0;
                } else {
                    correction_neg2 = (1.0 - cos_neg2_prev_curr)
                        * logistic((d_neg2_prev * t_ratio_neg2 - 1.5) * 4.0)
                        * 0.3;
                }
            } else {
                obj_prev_temp_in_mid = true;
                let normalized_pos_neg2 = s_neg2_prev / -t_neg2_prev * t_prev_curr;
                let x_neg2 = normalized_pos_neg2.dot(s_prev_curr) / d_prev_curr;
                let y_neg2 = (normalized_pos_neg2 - s_prev_curr * x_neg2 / d_prev_curr).length();

                // TODO
            }
        }

        todo!()
    }
}

#[inline]
fn calc_correction_0_stop(d: f32, x: f32, y: f32) -> f32 {
    logistic(10.0 * (x * x + y * y + 1.0).sqrt() - 12.0)
}

#[inline]
fn calculate_ip(d: f32, mt: f32) -> f32 {
    (d + 1.0).log2() / (mt + 1e-10)
}

#[inline]
fn interpolate(d: f32) -> f32 {
    let a = Pos2 { x: -1.0, y: 1.0 };
    let b = Pos2 { x: 1.1, y: 0.0 };

    (b.y - a.y) / (b.x - a.x) * (d - a.x) + a.y
}
