use super::{
    math_util::{fitts_ip, logistic},
    ArrayVec, OsuObject, FLOW_NEG2, FLOW_NEXT, SNAP_NEG2, SNAP_NEXT,
};

use rosu_pp::parse::Pos2;

const T_RATIO_THRESHOLD: f32 = 1.4;
const CORRECTION_NEG2_STILL: f32 = 0.0;

#[derive(Default)]
pub struct Movement {
    pub(crate) raw_movement_time: f32,
    pub(crate) dist: f32,
    pub(crate) movement_time: f32,
    pub(crate) index_of_perf: f32,
    pub(crate) cheesability: f32,
    pub(crate) cheesable_ratio: f32,
    pub(crate) time: f32,
    pub(crate) ends_on_slider: bool,
}

impl Movement {
    #[inline]
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

    pub(crate) fn extract_movement(h: &OsuObject) -> Vec<Self> {
        let init_time = h.time / 1000.0;
        let mut movement_with_nested = vec![Self::empty(init_time)];

        if h.nested_object_count > 0 {
            let extra_nested_count = h.nested_object_count - 1;

            for _ in 0..extra_nested_count {
                movement_with_nested.push(Self::empty(init_time));
            }
        }

        // println!(
        //     "init_time={} | count={}",
        //     init_time,
        //     movement_with_nested.len()
        // );

        movement_with_nested
    }

    pub(crate) fn extract_movement_complete(
        obj_neg2: Option<&OsuObject>,
        obj_prev: &OsuObject,
        obj_curr: &OsuObject,
        obj_next: Option<&OsuObject>,
        tap_strain: Option<&[f32; 4]>,
        clock_rate: f32,
        hidden: bool,
        note_density: Option<f32>,
        obj_neg4: Option<&OsuObject>,
        radius: f32,
    ) -> Vec<Self> {
        let note_density = note_density.unwrap_or(0.0);
        let movement_time = obj_curr.time / 1000.0;

        let mut movement = Self::default();

        let t_prev_curr = (obj_curr.time - obj_prev.time) / clock_rate / 1000.0;
        movement.raw_movement_time = t_prev_curr;
        movement.time = movement_time;

        if obj_curr.is_spinner() || obj_prev.is_spinner() {
            movement.movement_time = 1.0;

            return vec![movement];
        }

        let obj_neg2 = obj_neg2.filter(|h| !h.is_spinner());
        let obj_next = obj_next.filter(|h| !h.is_spinner());

        if obj_curr.is_slider() {
            movement.ends_on_slider = true;
        }

        let pos_prev = obj_prev.pos;
        let pos_curr = obj_curr.pos;
        let s_prev_curr = (pos_curr - pos_prev) / (2.0 * radius);

        let d_prev_curr = s_prev_curr.length();
        let ip_prev_curr = fitts_ip(d_prev_curr, t_prev_curr);

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

        // println!("pos_neg2={}", pos_neg2);
        // println!("pos_next={}", pos_next);
        // println!("s_neg2_prev={}", s_neg2_prev);
        // println!("s_curr_next={}", s_curr_next);
        // println!("d_neg2_prev={}", d_neg2_prev);
        // println!("d_neg2_curr={}", d_neg2_curr);
        // println!("d_curr_next={}", d_curr_next);
        // println!("t_neg2_prev={}", t_neg2_prev);
        // println!("t_curr_next={}", t_curr_next);
        // println!("flowiness_neg2_prev_curr={}", flowiness_neg2_prev_curr);
        // println!("flowiness_prev_curr_next={}", flowiness_prev_curr_next);
        // println!("obj_prev_temp_in_mid={}", obj_prev_temp_in_mid);
        // println!("obj_curr_temp_in_mid={}", obj_curr_temp_in_mid);

        // Correction #1
        let mut correction_neg2 = 0.0;

        if obj_neg2.filter(|_| d_prev_curr != 0.0).is_some() {
            let t_ratio_neg2 = t_prev_curr / t_neg2_prev;
            let cos_neg2_prev_curr = (s_neg2_prev.dot(s_prev_curr) / -d_neg2_prev / d_prev_curr)
                .max(-1.0)
                .min(1.0);

            // println!(
            //     "t_ratio_neg2={} | cos_neg2_prev_curr={}",
            //     t_ratio_neg2, cos_neg2_prev_curr
            // );

            if t_ratio_neg2 > T_RATIO_THRESHOLD {
                // println!("[a]");

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
                // println!("[b]");

                if d_neg2_prev.abs() < f32::EPSILON {
                    correction_neg2 = 0.0;
                } else {
                    correction_neg2 = (1.0 - cos_neg2_prev_curr)
                        * logistic((d_neg2_prev * t_ratio_neg2 - 1.5) * 4.0)
                        * 0.3;
                }
            } else {
                // println!("[c]");

                obj_prev_temp_in_mid = true;
                let normalized_pos_neg2 = s_neg2_prev / -t_neg2_prev * t_prev_curr;
                let x_neg2 = normalized_pos_neg2.dot(s_prev_curr) / d_prev_curr;
                let y_neg2 = (normalized_pos_neg2 - s_prev_curr * x_neg2 / d_prev_curr).length();

                // println!(
                //     "normalized={} | x_neg2={} | y_neg2={}",
                //     normalized_pos_neg2, x_neg2, y_neg2
                // );

                let correction_neg2_flow = FLOW_NEG2.evaluate(d_prev_curr, x_neg2, y_neg2);

                // println!("correction_neg2_flow={}", correction_neg2_flow);

                let correction_neg2_snap = SNAP_NEG2.evaluate(d_prev_curr, x_neg2, y_neg2);
                let correction_neg2_stop = calc_correction_0_stop(x_neg2, y_neg2);

                // println!(
                //     "flow={} | snap={} | stop={}",
                //     correction_neg2_flow, correction_neg2_snap, correction_neg2_stop
                // );

                flowiness_neg2_prev_curr =
                    logistic((correction_neg2_snap - correction_neg2_flow - 0.05) * 20.0);

                correction_neg2 = [
                    correction_neg2_flow,
                    correction_neg2_snap,
                    correction_neg2_stop,
                ]
                .powi_mean(-10)
                    * 1.3;
            }
        }

        // Correction #2
        let mut correction_next = 0.0;

        if obj_next.filter(|_| d_prev_curr != 0.0).is_some() {
            let t_ratio_next = t_prev_curr / t_curr_next;
            let cos_prev_curr_next = (s_prev_curr.dot(s_curr_next) / -d_prev_curr / d_curr_next)
                .min(1.0)
                .max(-1.0);

            correction_next = if t_ratio_next > T_RATIO_THRESHOLD {
                if d_curr_next.abs() < f32::EPSILON {
                    0.0
                } else {
                    let correction_next_moving = interpolate(cos_prev_curr_next);
                    let movingness = logistic(d_curr_next * 6.0 - 5.0) - logistic(-5.0);

                    movingness * correction_next_moving * 0.5
                }
            } else if t_ratio_next < T_RATIO_THRESHOLD.recip() {
                if d_curr_next.abs() < f32::EPSILON {
                    0.0
                } else {
                    (1.0 - cos_prev_curr_next)
                        * logistic((d_curr_next * t_ratio_next - 1.5) * 4.0)
                        * 0.15
                }
            } else {
                obj_curr_temp_in_mid = true;

                let normalized_pos_next = s_curr_next / t_curr_next * t_prev_curr;
                let x_next = normalized_pos_next.dot(s_prev_curr) / d_prev_curr;
                let y_next = (normalized_pos_next - s_prev_curr * x_next / d_prev_curr).length();

                let correction_next_flow = FLOW_NEXT.evaluate(d_prev_curr, x_next, y_next);
                let correction_next_snap = SNAP_NEXT.evaluate(d_prev_curr, x_next, y_next);

                flowiness_prev_curr_next =
                    logistic((correction_next_snap - correction_next_flow - 0.05) * 20.0);

                ([correction_next_flow, correction_next_snap].powi_mean(-10) - 0.1).max(0.0) * 0.5
            };
        }

        // Correction #3
        let mut pattern_correction = 0.0; // TODO

        if obj_prev_temp_in_mid && obj_curr_temp_in_mid {
            let gap = (s_prev_curr - s_curr_next / 2.0 - s_neg2_prev / 2.0).length()
                / (d_prev_curr + 0.1);

            // println!(
            //     "flowiness_neg2={} | flowiness_next={}",
            //     flowiness_neg2_prev_curr, flowiness_prev_curr_next
            // );

            pattern_correction = (logistic((gap - 1.0) * 8.0) - logistic(-6.0))
                * logistic((d_neg2_prev - 0.7) * 10.0)
                * logistic((d_curr_next - 0.7) * 10.0)
                * [flowiness_neg2_prev_curr, flowiness_prev_curr_next].powi_mean(2)
                * 0.6;
        }

        // Correction #4
        let mut tap_correction = 0.0;

        if let Some(tap_strain) = tap_strain.filter(|_| d_prev_curr > 0.0) {
            tap_correction = logistic((tap_strain.powi_mean(2) / ip_prev_curr - 1.34) / 0.1) * 0.15;
        }

        // Correction #5
        let mut time_early = 0.0;
        let mut time_late = 0.0;
        let mut cheesability_early = 0.0;
        let mut cheesability_late = 0.0;

        if d_prev_curr > 0.0 {
            let mut t_neg2_prev_recip = 0.0;
            let mut ip_neg2_prev = 0.0;

            if obj_neg2.is_some() {
                t_neg2_prev_recip = (t_neg2_prev + 1e-10).recip();
                ip_neg2_prev = fitts_ip(d_neg2_prev, t_neg2_prev);
            }

            cheesability_early = logistic((ip_neg2_prev / ip_prev_curr - 0.6) * -15.0) * 0.5;
            time_early =
                cheesability_early * (1.0 / (1.0 / (t_prev_curr + 0.07) + t_neg2_prev_recip));

            let mut t_curr_next_recip = 0.0;
            let mut ip_curr_next = 0.0;

            if obj_next.is_some() {
                t_curr_next_recip = (t_curr_next + 1e-10).recip();
                ip_curr_next = fitts_ip(d_curr_next, t_curr_next);
            }

            cheesability_late = logistic((ip_curr_next / ip_prev_curr - 0.6) * -15.0) * 0.5;
            time_late =
                cheesability_late * (1.0 / (1.0 / (t_prev_curr + 0.07) + t_curr_next_recip));
        }

        // Correction #6
        let effective_bpm = 30.0 / (t_prev_curr + 1e-10);
        let high_bpm_jump_buff =
            logistic((effective_bpm - 354.0) / 16.0) * logistic((d_prev_curr - 1.9) / 0.15) * 0.23;

        // Correction #7
        let small_circle_bonus = ((logistic((55.0 - 2.0 * radius) / 2.9) * 0.275)
            + (logistic((-radius + 10.0) / 4.0) * 0.8).min(24.5))
            * logistic((d_prev_curr - 0.5) / 0.1).max(0.25);

        // Correction #8
        let d_prev_curr_stacked_nerf = (1.2 * d_prev_curr - 0.185)
            .min(1.4 * d_prev_curr - 0.32)
            .max(0.0)
            .min(d_prev_curr);

        // Correction #9
        let small_jump_nerf_factor = 1.0
            - 0.17
                * (-((d_prev_curr - 2.2) / 0.7).powi(2)).exp()
                * logistic((255.0 - effective_bpm) / 10.0);

        // Correction #10
        let big_jump_buff_factor = 1.0
            + 0.15 * logistic((d_prev_curr - 6.0) / 0.5) * logistic((210.0 - effective_bpm) / 8.0);

        // Correction #11
        let correction_hidden = hidden as u8 as f32 * (0.05 + 0.008 * note_density);

        // Correction #12
        if obj_neg2.is_some() && obj_next.is_some() {
            let d_prev_next = ((pos_next - pos_prev) / (2.0 * radius)).length();
            let d_neg2_next = ((pos_next - pos_neg2) / (2.0 * radius)).length();

            if d_neg2_prev < 1.0
                && d_neg2_curr < 1.0
                && d_neg2_next < 1.0
                && d_prev_curr < 1.0
                && d_prev_next < 1.0
                && d_curr_next < 1.0
            {
                correction_neg2 = 0.0;
                correction_next = 0.0;
                pattern_correction = 0.0;
                tap_correction = 0.0;
            }
        }

        // Correction #13
        let jump_overlap_correction = 1.0
            - ((0.15 - 0.1 * d_neg2_curr).max(0.0) + (0.1125 - 0.075 * d_neg4_curr).max(0.0))
                * logistic((d_prev_curr - 3.3) / 0.25);

        // Correction #14
        let dist_increase_buff = 1.0;

        // Apply all corrections
        let d_prev_curr_with_correction = d_prev_curr_stacked_nerf
            * (1.0 + small_circle_bonus)
            * (1.0 + correction_neg2 + correction_next + pattern_correction)
            * (1.0 + high_bpm_jump_buff)
            * (1.0 + tap_correction)
            * small_jump_nerf_factor
            * big_jump_buff_factor
            * (1.0 + correction_hidden)
            * jump_overlap_correction
            * dist_increase_buff;

        // println!("small_circle_bonus={}", small_circle_bonus);
        // println!(
        //     "neg2 + next + pattern: {} + {} + {}",
        //     correction_neg2, correction_next, pattern_correction
        // );
        // println!("high_bpm_jump_buff={}", high_bpm_jump_buff);
        // println!("tap_correction={}", tap_correction);
        // println!("small_jump_nerf_factor={}", small_jump_nerf_factor);
        // println!("big_jump_buff_factor={}", big_jump_buff_factor);
        // println!("correction_hidden={}", correction_hidden);
        // println!("jump_overlap_correction={}", jump_overlap_correction);
        // println!(" ==> {}", d_prev_curr_with_correction);

        movement.dist = d_prev_curr_with_correction;
        movement.movement_time = t_prev_curr;
        movement.cheesability = cheesability_early + cheesability_late;
        movement.cheesable_ratio = (time_early + time_late) / (t_prev_curr + 1e-10);

        let mut movement_with_nested = vec![movement];

        let extra_nested_count = obj_curr.nested_object_count.saturating_sub(1);

        for _ in 0..extra_nested_count {
            movement_with_nested.push(Movement::empty(movement_time));
        }

        movement_with_nested
    }
}

#[inline]
fn calc_correction_0_stop(x: f32, y: f32) -> f32 {
    logistic(10.0 * (x * x + y * y + 1.0).sqrt() - 12.0)
}

#[inline]
fn interpolate(d: f32) -> f32 {
    let a = Pos2 { x: -1.0, y: 1.0 };
    let b = Pos2 { x: 1.1, y: 0.0 };

    (b.y - a.y) / (b.x - a.x) * (d - a.x) + a.y
}
