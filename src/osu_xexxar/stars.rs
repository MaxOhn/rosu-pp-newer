//! The positional offset of notes created by stack leniency is not considered.
//! This means the jump distance inbetween notes might be slightly off, resulting in small inaccuracies.
//! Since calculating these offsets is relatively expensive though, this version is faster than `all_included`.

use super::{difficulty_range_od, DifficultyObject, OsuObject, Skill, SkillKind, SliderState};

use rosu_pp::{osu::DifficultyAttributes, Beatmap, Mods, StarResult};

const OBJECT_RADIUS: f32 = 64.0;
const DIFFICULTY_MULTIPLIER: f32 = 0.18;
const NORMALIZED_RADIUS: f32 = 50.0;
const DISPLAY_DIFFICULTY_MULTIPLIER: f32 = 0.605;

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

    let mut aim = Skill::new(SkillKind::Aim);
    let mut tap = Skill::new(SkillKind::Tap);

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
        aim.calculate_display_value().powf(0.75) * DISPLAY_DIFFICULTY_MULTIPLIER;
    let display_speed_rating =
        tap.calculate_display_value().powf(0.75) * DISPLAY_DIFFICULTY_MULTIPLIER;

    let display_aim_perf = (display_aim_rating.max(1.0) * 5.0 - 4.0).powi(3) / 100_000.0;
    let display_speed_perf = (display_speed_rating.max(1.0) * 5.0 - 4.0).powi(3) / 100_000.0;

    let total_pp = (display_aim_perf.powf(1.1) + display_speed_perf.powf(1.1)).powf(1.0 / 1.1);

    let stars = 0.027 * ((100_000.0 / (1.0_f32 / 1.1).exp2() * total_pp).cbrt() + 4.0);

    diff_attributes.n_circles = map.n_circles as usize;
    diff_attributes.n_spinners = map.n_spinners as usize;
    diff_attributes.stars = stars;
    diff_attributes.speed_strain = speed_rating;
    diff_attributes.aim_strain = aim_rating;

    StarResult::Osu(diff_attributes)
}
