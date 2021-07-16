//! The positional offset of notes created by stack leniency is not considered.
//! This means the jump distance inbetween notes might be slightly off, resulting in small inaccuracies.
//! Since calculating these offsets is relatively expensive though, this version is faster than `all_included`.

use super::{
    difficulty_range_ar, difficulty_range_od, Beatmap, DifficultyObject, OsuObject, Skill,
    SkillKind, SliderState,
};

use rosu_pp::{osu::DifficultyAttributes, parse::Pos2, Mods, StarResult};

const OBJECT_RADIUS: f32 = 64.0;
const DIFFICULTY_MULTIPLIER: f32 = 0.18;
const NORMALIZED_RADIUS: f32 = 50.0;
const DISPLAY_DIFFICULTY_MULTIPLIER: f32 = 0.04;
const STACK_DISTANCE: f32 = 3.0;

/// Star calculation for osu!standard maps.
///
/// Slider paths are considered but stack leniency is ignored.
/// As most maps don't even make use of leniency and even if,
/// it has generally little effect on stars, the results are close to perfect.
/// This version is considerably more efficient than `all_included` since
/// processing stack leniency is relatively expensive.
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

    let scale = (1.0 - 0.7 * (map_attributes.cs - 5.0) / 5.0) / 2.0;
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

    let mut hit_objects = Vec::with_capacity(map.hit_objects.len());
    hit_objects.extend(hit_objects_iter);

    let mut raw_ar = map.ar;
    let hr = mods.hr();

    if hr {
        raw_ar = (raw_ar * 1.4).min(10.0);
    } else if mods.ez() {
        raw_ar *= 0.5;
    }

    let time_preempt = difficulty_range_ar(raw_ar);
    let stack_threshold = time_preempt * map.stack_leniency;

    if map.version >= 6 {
        stacking(&mut hit_objects, stack_threshold);
    } else {
        old_stacking(&mut hit_objects, stack_threshold);
    }

    hit_objects.iter_mut().for_each(|h| {
        let stack_offset = h.stack_height * scale * -6.4;

        h.pos = if hr {
            Pos2 {
                x: h.pos.x + stack_offset,
                y: 384.0 - ((384.0 - h.pos.y) + stack_offset),
            }
        } else {
            h.pos
                + Pos2 {
                    x: stack_offset,
                    y: stack_offset,
                }
        };
    });

    let mut aim = Skill::new(SkillKind::new_aim());
    let mut tap = Skill::new(SkillKind::new_tap());

    // First object has no predecessor and thus no strain, handle distinctly
    let mut prev_prev = None;
    let mut prev = &hit_objects[0];
    let mut prev_vals = None;

    // Handle second object separately to remove later if-branching
    let curr = &hit_objects[1];
    let h = DifficultyObject::new(
        &curr,
        &prev,
        prev_vals,
        prev_prev,
        map_attributes.clock_rate,
        scaling_factor,
    );

    prev_prev.replace(prev);
    prev_vals.replace((h.jump_dist, h.strain_time));
    prev = curr;

    aim.process_internal(h.clone());
    tap.process_internal(h);

    // Handle all other objects
    for curr in hit_objects.iter().skip(2) {
        let h = DifficultyObject::new(
            curr,
            prev,
            prev_vals,
            prev_prev,
            map_attributes.clock_rate,
            scaling_factor,
        );

        prev_prev.replace(prev);
        prev_vals.replace((h.jump_dist, h.strain_time));
        prev = curr;

        aim.process_internal(h.clone());
        tap.process_internal(h);
    }

    let aim_rating = aim.difficulty_value().powf(0.75) * DIFFICULTY_MULTIPLIER;
    let speed_rating = tap.difficulty_value().powf(0.75) * DIFFICULTY_MULTIPLIER;

    let display_aim_rating =
        aim.display_difficulty_value().powf(0.75) * DISPLAY_DIFFICULTY_MULTIPLIER;
    let display_speed_rating =
        tap.display_difficulty_value().powf(0.75) * DISPLAY_DIFFICULTY_MULTIPLIER;

    let stars = display_aim_rating + display_speed_rating;

    diff_attributes.n_circles = map.n_circles as usize;
    diff_attributes.n_spinners = map.n_spinners as usize;
    diff_attributes.stars = stars;
    diff_attributes.speed_strain = speed_rating;
    diff_attributes.aim_strain = aim_rating;

    StarResult::Osu(diff_attributes)
}

fn stacking(hit_objects: &mut [OsuObject], stack_threshold: f32) {
    let mut extended_start_idx = 0;
    let extended_end_idx = hit_objects.len() - 1;

    for mut i in (1..=extended_end_idx).rev() {
        let mut n = i;

        if hit_objects[i].stack_height != 0.0 || hit_objects[i].is_spinner() {
            continue;
        }

        if hit_objects[i].is_circle() {
            loop {
                n = match n.checked_sub(1) {
                    Some(n) => n,
                    None => break,
                };

                if hit_objects[n].is_spinner() {
                    continue;
                } else if hit_objects[i].time - hit_objects[n].end_time() > stack_threshold {
                    break;
                } else if n < extended_start_idx {
                    hit_objects[n].stack_height = 0.0;
                    extended_start_idx = n;
                }

                if hit_objects[n].is_slider()
                    && hit_objects[n].end_pos().distance(hit_objects[i].pos) < STACK_DISTANCE
                {
                    let offset = hit_objects[i].stack_height - hit_objects[n].stack_height + 1.0;

                    for j in n + 1..=i {
                        if hit_objects[n].pos.distance(hit_objects[j].pos) < STACK_DISTANCE {
                            hit_objects[j].stack_height -= offset;
                        }
                    }

                    break;
                }

                if hit_objects[n].pos.distance(hit_objects[i].pos) < STACK_DISTANCE {
                    hit_objects[n].stack_height = hit_objects[i].stack_height + 1.0;
                    i = n;
                }
            }
        } else if hit_objects[i].is_slider() {
            loop {
                n = match n.checked_sub(1) {
                    Some(n) => n,
                    None => break,
                };

                if hit_objects[n].is_spinner() {
                    continue;
                } else if hit_objects[i].time - hit_objects[n].time > stack_threshold {
                    break;
                } else if hit_objects[n].end_pos().distance(hit_objects[i].pos) < STACK_DISTANCE {
                    hit_objects[n].stack_height = hit_objects[i].stack_height + 1.0;
                    i = n;
                }
            }
        }
    }
}

fn old_stacking(hit_objects: &mut [OsuObject], stack_threshold: f32) {
    for i in 0..hit_objects.len() {
        if hit_objects[i].stack_height != 0.0 && !hit_objects[i].is_slider() {
            continue;
        }

        let mut start_time = hit_objects[i].end_time();
        let end_pos = hit_objects[i].end_pos();

        let mut slider_stack = 0.0;

        for j in i + 1..hit_objects.len() {
            if hit_objects[j].time - stack_threshold > start_time {
                break;
            }

            if hit_objects[j].pos.distance(hit_objects[i].pos) < STACK_DISTANCE {
                hit_objects[i].stack_height += 1.0;
                start_time = hit_objects[j].end_time();
            } else if hit_objects[j].pos.distance(end_pos) < STACK_DISTANCE {
                slider_stack += 1.0;
                hit_objects[j].stack_height -= slider_stack;
                start_time = hit_objects[j].end_time();
            }
        }
    }
}
