mod angle_correction;
use angle_correction::*;

mod array_vec;
use array_vec::ArrayVec;

mod control_point_iter;
use control_point_iter::{ControlPoint, ControlPointIter};

mod hit_probabilities;
use hit_probabilities::HitProbabilities;

mod interpolations;
use interpolations::{CubicInterpolation, TricubicInterpolation};

mod math_util;

mod movement;
use movement::Movement;

mod osu_object;
use osu_object::OsuObject;

mod pp;
pub use pp::{OsuAttributeProvider, OsuPP};

mod slider_state;
use slider_state::SliderState;

mod stars;
pub use stars::stars;

#[derive(Clone, Debug, Default)]
pub struct DifficultyAttributes {
    stars: f32,

    tap_sr: f32,
    tap_diff: f32,
    stream_note_count: f32,
    mash_tap_diff: f32,

    finger_control_sr: f32,
    finger_control_diff: f32,

    aim_sr: f32,
    aim_diff: f32,
    aim_hidden_factor: f32,
    combo_tps: Vec<f32>,
    miss_tps: Vec<f32>,
    miss_counts: Vec<f32>,
    cheese_note_count: f32,
    cheese_levels: Vec<f32>,
    cheese_factors: Vec<f32>,

    length: f32,
    ar: f32,
    od: f32,
    max_combo: usize,

    n_circles: u32,
    n_sliders: u32,
    n_spinners: u32,
}

#[inline]
fn difficulty_range(val: f32, max: f32, avg: f32, min: f32) -> f32 {
    if val > 5.0 {
        avg + (max - avg) * (val - 5.0) / 5.0
    } else if val < 5.0 {
        avg - (avg - min) * (5.0 - val) / 5.0
    } else {
        avg
    }
}

const OSU_OD_MAX: f32 = 20.0;
const OSU_OD_AVG: f32 = 50.0;
const OSU_OD_MIN: f32 = 80.0;

const OSU_AR_MAX: f32 = 450.0;
const OSU_AR_AVG: f32 = 1200.0;
const OSU_AR_MIN: f32 = 1800.0;

#[inline]
fn difficulty_range_od(od: f32) -> f32 {
    difficulty_range(od, OSU_OD_MAX, OSU_OD_AVG, OSU_OD_MIN)
}

#[inline]
fn difficulty_range_ar(ar: f32) -> f32 {
    difficulty_range(ar, OSU_AR_MAX, OSU_AR_AVG, OSU_AR_MIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosu_pp::Beatmap;
    use tokio::fs::File;

    #[tokio::test]
    async fn single() {
        let map_id = 786018;

        let file = File::open(format!(
            "C:/Users/Max/Desktop/Coding/C#/osu-tools/cache/{}.osu",
            map_id
        ))
        .await
        .unwrap();
        let map = Beatmap::parse(file).await.unwrap();

        let result = OsuPP::new(&map)
            // .n300(1184)
            // .n100(32)
            // .n50(0)
            // .misses(0)
            // .combo(1588)
            .mods(0)
            .calculate();

        println!("Stars={} | PP={}", result.stars(), result.pp());
    }
}
