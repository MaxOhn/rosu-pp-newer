use super::{DifficultyAttributes, SliderState};

use rosu_pp::{
    parse::{HitObject, HitObjectKind, Pos2},
    Beatmap,
};

pub(crate) struct OsuObject {
    pub(crate) time: f32,
    pub(crate) pos: Pos2,
    kind: ObjectKind,
    pub(crate) nested_object_count: usize,
}

enum ObjectKind {
    Circle,
    Slider,
    Spinner,
}

impl OsuObject {
    pub(crate) fn new(
        h: &HitObject,
        map: &Beatmap,
        ticks: &mut Vec<f32>,
        attributes: &mut DifficultyAttributes,
        slider_state: &mut SliderState,
    ) -> Option<Self> {
        attributes.max_combo += 1; // hitcircle, slider head, or spinner

        let obj = match &h.kind {
            HitObjectKind::Circle => Self {
                time: h.start_time,
                pos: h.pos,
                kind: ObjectKind::Circle,
                nested_object_count: 0,
            },
            HitObjectKind::Slider {
                pixel_len, repeats, ..
            } => {
                // Key values which are computed here
                let mut nested_object_count = 1;

                // Responsible for timing point values
                slider_state.update(h.start_time);

                let mut tick_distance = 100.0 * map.sv / map.tick_rate;

                if map.version >= 8 {
                    tick_distance /=
                        (100.0 / slider_state.speed_mult).max(10.0).min(1000.0) / 100.0;
                }

                let duration = *repeats as f32 * slider_state.beat_len * pixel_len
                    / (map.sv * slider_state.speed_mult)
                    / 100.0;

                // Called on each slider object except for the head.
                // Increases combo and adjusts `end_pos` and `travel_dist`
                // w.r.t. the object position at the given time on the slider curve.
                let mut compute_vertex = || {
                    attributes.max_combo += 1;
                    nested_object_count += 1;
                };

                let mut current_distance = tick_distance;
                let time_add = duration * (tick_distance / (pixel_len * *repeats as f32));

                let target = pixel_len - tick_distance / 8.0;
                ticks.reserve((target / tick_distance) as usize);

                // Tick of the first span
                if current_distance < target {
                    for tick_idx in 1.. {
                        let time = h.start_time + time_add * tick_idx as f32;
                        compute_vertex();
                        ticks.push(time);
                        current_distance += tick_distance;

                        if current_distance >= target {
                            break;
                        }
                    }
                }

                // Other spans
                if *repeats > 1 {
                    for repeat_id in 1..*repeats {
                        // Reverse tick
                        compute_vertex();

                        // Actual ticks
                        if repeat_id & 1 == 1 {
                            ticks.iter().rev().for_each(|_| compute_vertex());
                        } else {
                            ticks.iter().for_each(|_| compute_vertex());
                        }
                    }
                }

                // Slider tail
                compute_vertex();

                ticks.clear();

                Self {
                    time: h.start_time,
                    pos: h.pos,
                    kind: ObjectKind::Slider,
                    nested_object_count,
                }
            }
            HitObjectKind::Spinner { .. } => Self {
                time: h.start_time,
                pos: h.pos,
                kind: ObjectKind::Spinner,
                nested_object_count: 0,
            },
            HitObjectKind::Hold { .. } => return None,
        };

        Some(obj)
    }

    #[inline]
    pub(crate) fn is_slider(&self) -> bool {
        matches!(self.kind, ObjectKind::Slider)
    }

    #[inline]
    pub(crate) fn is_spinner(&self) -> bool {
        matches!(self.kind, ObjectKind::Spinner)
    }
}
