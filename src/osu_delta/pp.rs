use std::f32::consts::{FRAC_PI_2, PI, SQRT_2};

use crate::osu_delta::math_util::{logistic, pow_mean};

use super::{
    array_vec::ArrayVec,
    math_util::{erfinv, linear_spaced},
    stars, CubicInterpolation, DifficultyAttributes,
};
use rosu_pp::{Beatmap, Mods};

#[derive(Clone, Debug)]
pub struct PpResult {
    pub pp: f32,
    pub attributes: DifficultyAttributes,
}

impl PpResult {
    #[inline]
    pub fn pp(&self) -> f32 {
        self.pp
    }

    #[inline]
    pub fn stars(&self) -> f32 {
        self.attributes.stars
    }
}

const TOTAL_VALUE_EXPONENT: f32 = 1.5;
const SKILL_TO_PP_EXPONENT: f32 = 2.7;
const MISS_COUNT_LENIENCY: f32 = 0.5;

/// Calculator for pp on osu!standard maps.
///
/// # Example
///
/// ```
/// # use rosu_pp::{OsuPP, PpResult, Beatmap};
/// # /*
/// let map: Beatmap = ...
/// # */
/// # let map = Beatmap::default();
/// let pp_result: PpResult = OsuPP::new(&map)
///     .mods(8 + 64) // HDDT
///     .combo(1234)
///     .misses(1)
///     .accuracy(98.5) // should be set last
///     .calculate();
///
/// println!("PP: {} | Stars: {}", pp_result.pp(), pp_result.stars());
///
/// let next_result = OsuPP::new(&map)
///     .attributes(pp_result)  // reusing previous results for performance
///     .mods(8 + 64)           // has to be the same to reuse attributes
///     .accuracy(99.5)
///     .calculate();
///
/// println!("PP: {} | Stars: {}", next_result.pp(), next_result.stars());
/// ```
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct OsuPP<'m> {
    map: &'m Beatmap,
    attributes: Option<DifficultyAttributes>,
    mods: u32,
    combo: Option<usize>,
    acc: Option<f32>,

    n300: Option<usize>,
    n100: Option<usize>,
    n50: Option<usize>,
    n_misses: usize,
    passed_objects: Option<usize>,
}

impl<'m> OsuPP<'m> {
    #[inline]
    pub fn new(map: &'m Beatmap) -> Self {
        Self {
            map,
            attributes: None,
            mods: 0,
            combo: None,
            acc: None,

            n300: None,
            n100: None,
            n50: None,
            n_misses: 0,
            passed_objects: None,
        }
    }

    /// [`OsuAttributeProvider`] is implemented by [`DifficultyAttributes`](crate::osu::DifficultyAttributes)
    /// and by [`PpResult`](crate::PpResult) meaning you can give the
    /// result of a star calculation or a pp calculation.
    /// If you already calculated the attributes for the current map-mod combination,
    /// be sure to put them in here so that they don't have to be recalculated.
    #[inline]
    pub fn attributes(mut self, attributes: impl OsuAttributeProvider) -> Self {
        self.attributes.replace(attributes.attributes());

        self
    }

    /// Specify mods through their bit values.
    ///
    /// See [https://github.com/ppy/osu-api/wiki#mods](https://github.com/ppy/osu-api/wiki#mods)
    #[inline]
    pub fn mods(mut self, mods: u32) -> Self {
        self.mods = mods;

        self
    }

    /// Specify the max combo of the play.
    #[inline]
    pub fn combo(mut self, combo: usize) -> Self {
        self.combo.replace(combo);

        self
    }

    /// Specify the amount of 300s of a play.
    #[inline]
    pub fn n300(mut self, n300: usize) -> Self {
        self.n300.replace(n300);

        self
    }

    /// Specify the amount of 100s of a play.
    #[inline]
    pub fn n100(mut self, n100: usize) -> Self {
        self.n100.replace(n100);

        self
    }

    /// Specify the amount of 50s of a play.
    #[inline]
    pub fn n50(mut self, n50: usize) -> Self {
        self.n50.replace(n50);

        self
    }

    /// Specify the amount of misses of a play.
    #[inline]
    pub fn misses(mut self, n_misses: usize) -> Self {
        self.n_misses = n_misses;

        self
    }

    /// Amount of passed objects for partial plays, e.g. a fail.
    #[inline]
    pub fn passed_objects(mut self, passed_objects: usize) -> Self {
        self.passed_objects.replace(passed_objects);

        self
    }

    /// Generate the hit results with respect to the given accuracy between `0` and `100`.
    ///
    /// Be sure to set `misses` beforehand!
    /// In case of a partial play, be also sure to set `passed_objects` beforehand!
    pub fn accuracy(mut self, acc: f32) -> Self {
        let n_objects = self
            .passed_objects
            .unwrap_or_else(|| self.map.hit_objects.len());

        let acc = acc / 100.0;

        if self.n100.or(self.n50).is_some() {
            let mut n100 = self.n100.unwrap_or(0);
            let mut n50 = self.n50.unwrap_or(0);

            let placed_points = 2 * n100 + n50 + self.n_misses;
            let missing_objects = n_objects - n100 - n50 - self.n_misses;
            let missing_points =
                ((6.0 * acc * n_objects as f32).round() as usize).saturating_sub(placed_points);

            let mut n300 = missing_objects.min(missing_points / 6);
            n50 += missing_objects - n300;

            if let Some(orig_n50) = self.n50.filter(|_| self.n100.is_none()) {
                // Only n50s were changed, try to load some off again onto n100s
                let difference = n50 - orig_n50;
                let n = n300.min(difference / 4);

                n300 -= n;
                n100 += 5 * n;
                n50 -= 4 * n;
            }

            self.n300.replace(n300);
            self.n100.replace(n100);
            self.n50.replace(n50);
        } else {
            let misses = self.n_misses.min(n_objects);
            let target_total = (acc * n_objects as f32 * 6.0).round() as usize;
            let delta = target_total - (n_objects - misses);

            let mut n300 = delta / 5;
            let mut n100 = (delta % 5).min(n_objects - n300 - misses);
            let mut n50 = n_objects - n300 - n100 - misses;

            // Sacrifice n300s to transform n50s into n100s
            let n = n300.min(n50 / 4);
            n300 -= n;
            n100 += 5 * n;
            n50 -= 4 * n;

            self.n300.replace(n300);
            self.n100.replace(n100);
            self.n50.replace(n50);
        }

        let acc = (6 * self.n300.unwrap() + 2 * self.n100.unwrap() + self.n50.unwrap()) as f32
            / (6 * n_objects) as f32;

        self.acc.replace(acc);

        self
    }

    fn assert_hitresults(&mut self) {
        if self.acc.is_none() {
            let n_objects = self
                .passed_objects
                .unwrap_or_else(|| self.map.hit_objects.len());

            let remaining = n_objects
                .saturating_sub(self.n300.unwrap_or(0))
                .saturating_sub(self.n100.unwrap_or(0))
                .saturating_sub(self.n50.unwrap_or(0))
                .saturating_sub(self.n_misses);

            if remaining > 0 {
                if self.n300.is_none() {
                    self.n300.replace(remaining);
                } else if self.n100.is_none() {
                    self.n100.replace(remaining);
                } else if self.n50.is_none() {
                    self.n50.replace(remaining);
                } else {
                    *self.n300.as_mut().unwrap() += remaining;
                }
            }

            let n300 = *self.n300.get_or_insert(0);
            let n100 = *self.n100.get_or_insert(0);
            let n50 = *self.n50.get_or_insert(0);

            let numerator = n300 * 6 + n100 * 2 + n50;
            self.acc.replace(numerator as f32 / n_objects as f32 / 6.0);
        }
    }

    pub fn calculate(mut self) -> PpResult {
        let attributes = match self.attributes.take() {
            Some(attributes) => attributes,
            None => stars(self.map, self.mods, self.passed_objects),
        };

        let great_window = 79.5 - 6.0 * attributes.od;

        // Make sure the hitresults and accuracy are set
        self.assert_hitresults();

        let mut multiplier = 2.14;

        let combo_based_miss_count = if attributes.n_sliders == 0 {
            if let Some(combo) = self.combo.filter(|&combo| combo < attributes.max_combo) {
                attributes.max_combo as f32 / combo as f32
            } else {
                0.0
            }
        } else {
            let full_combo_threshold =
                attributes.max_combo as f32 - 0.1 * attributes.n_sliders as f32;

            if let Some(combo) = self
                .combo
                .filter(|&combo| (combo as f32) < full_combo_threshold)
            {
                full_combo_threshold as f32 / combo as f32
            } else {
                ((attributes.max_combo - self.combo.unwrap_or(attributes.max_combo)) as f32
                    / (0.1 * attributes.n_sliders as f32))
                    .powi(3)
            }
        };

        let effective_miss_count = combo_based_miss_count.max(self.n_misses as f32);
        let total_hits = self.total_hits() as f32;

        // NF penalty
        if self.mods.nf() {
            multiplier *= (1.0 - 0.02 * self.n_misses as f32).max(0.9);
        }

        // SO penalty
        if self.mods.so() {
            let n_spinners = attributes.n_spinners;
            multiplier *= 1.0 - (n_spinners as f32 / total_hits).powf(0.85);
        }

        let aim_value =
            self.compute_aim_value(total_hits, &attributes, great_window, effective_miss_count);
        let tap_value =
            self.compute_tap_value(total_hits, &attributes, great_window, effective_miss_count);
        let acc_value = self.compute_acc_value(&attributes, great_window, effective_miss_count);

        let pp = [aim_value, tap_value, acc_value].powf_mean(TOTAL_VALUE_EXPONENT) * multiplier;

        PpResult { pp, attributes }
    }

    fn compute_aim_value(
        &self,
        total_hits: f32,
        attributes: &DifficultyAttributes,
        great_window: f32,
        effective_miss_count: f32,
    ) -> f32 {
        if total_hits <= 1.0 {
            return 0.0;
        }

        let combo_tp_count = attributes.combo_tps.len();
        let combo_percentage = linear_spaced(combo_tp_count, (combo_tp_count as f32).recip(), 1.0);

        let max_combo = attributes.max_combo;
        let combo = self.combo.unwrap_or(max_combo);

        let score_combo_percentage = combo as f32 / max_combo as f32;
        let combo_tp =
            CubicInterpolation::new(&combo_percentage, &attributes.combo_tps, None, None)
                .evaluate(score_combo_percentage);

        let mut miss_tp =
            CubicInterpolation::new(&attributes.miss_counts, &attributes.miss_tps, None, None)
                .evaluate(effective_miss_count);

        miss_tp = miss_tp.max(0.0);

        let mut tp = pow_mean(combo_tp, miss_tp, 20.0);

        if self.mods.hd() {
            let hidden_factor = if attributes.ar > 10.75 {
                1.0
            } else if attributes.ar > 9.75 {
                1.0 + (1.0 - (((attributes.ar - 9.75) * FRAC_PI_2).sin()).powi(2))
                    * (attributes.aim_hidden_factor - 1.0)
            } else {
                attributes.aim_hidden_factor
            };

            tp *= hidden_factor;
        }

        // Account for cheesing
        let modified_acc = self.modified_acc(attributes);
        let acc_on_cheese_notes =
            1.0 - (1.0 - modified_acc) * (total_hits / attributes.cheese_note_count).sqrt();

        let acc_on_cheese_notes_positive = (acc_on_cheese_notes - 1.0).exp();

        let ur_on_cheese_notes =
            10.0 * great_window / (SQRT_2 * erfinv(acc_on_cheese_notes_positive));
        let cheese_level = logistic(((ur_on_cheese_notes * attributes.aim_diff) - 3200.0) / 2000.0);

        let cheese_factor = CubicInterpolation::new(
            &attributes.cheese_levels,
            &attributes.cheese_factors,
            None,
            None,
        )
        .evaluate(cheese_level);

        if self.mods.td() {
            tp = tp.min(1.47 * tp.powf(0.8));
        }

        let mut aim_value = tp_to_pp(tp * cheese_factor);

        // penalize misses
        aim_value *= 0.96_f32.powf((effective_miss_count - MISS_COUNT_LENIENCY).max(0.0));

        // Buff long maps
        aim_value *=
            1.0 + (logistic((total_hits - 2800.0) / 500.0) - logistic(-2800.0 / 500.0)) * 0.22;

        // Buff very high AR and low AR
        let mut ar_factor = 1.0;

        if attributes.ar > 10.0 {
            ar_factor += (0.05 + 0.35 * ((PI * total_hits.min(1250.0) / 2500.0).sin()).powf(1.7))
                * (attributes.ar - 10.0).powi(2);
        } else if attributes.ar < 8.0 {
            ar_factor += 0.01 * (8.0 - attributes.ar);
        }

        aim_value *= ar_factor;

        if self.mods.fl() {
            aim_value *= 1.0
                + 0.35 * (total_hits / 200.0).min(1.0)
                + (total_hits > 200.0) as u8 as f32
                    * (0.3 * ((total_hits - 200.0) / 300.0).min(1.0)
                        + (total_hits > 500.0) as u8 as f32 * (total_hits - 500.0) / 2000.0);
        }

        // Scale the aim value down with accuracy
        let acc_leniency = great_window * attributes.aim_diff / 300.0;
        let acc_penalty = (0.09 / (self.acc.unwrap_or(1.0) - 1.3) + 0.3) * (acc_leniency + 1.5);

        aim_value *= 0.2 + logistic(-((acc_penalty - 0.24953) / 0.18));

        aim_value
    }

    fn compute_tap_value(
        &self,
        total_hits: f32,
        attributes: &DifficultyAttributes,
        great_window: f32,
        effective_miss_count: f32,
    ) -> f32 {
        if total_hits <= 1.0 {
            return 0.0;
        }

        let modified_acc = self.modified_acc(attributes);

        let acc_on_stream =
            1.0 - (1.0 - modified_acc) * (total_hits / attributes.stream_note_count).sqrt();

        let acc_on_streams_positive = (acc_on_stream - 1.0).exp();
        let ur_on_streams = 10.0 * great_window / (SQRT_2 * erfinv(acc_on_streams_positive));
        let mash_level = logistic(((ur_on_streams * attributes.tap_diff) - 4000.0) / 1000.0);

        let tap_skill =
            mash_level * attributes.mash_tap_diff + (1.0 - mash_level) * attributes.tap_diff;

        let mut tap_value = tap_skill_to_pp(tap_skill);

        // Buff very high acc on streams
        let acc_buff = ((acc_on_stream - 1.0) * 60.0).exp() * tap_value * 0.2;
        tap_value += acc_buff;

        // Scale tap value down with accuracy
        let od_scale = logistic(16.0 - great_window) * 0.04;
        let acc = self.acc.unwrap_or(1.0);
        let acc_factor = 0.5
            + 0.5
                * ((logistic((acc - 0.9543 + 1.83 * od_scale) / 0.025 + od_scale)).powf(0.2)
                    + logistic(-3.5));

        tap_value *= acc_factor;

        // Penalize misses and 50s exponentially
        tap_value *= 0.93_f32.powf((effective_miss_count - MISS_COUNT_LENIENCY).max(0.0));

        let n50 = self.n50.unwrap_or(0) as f32;
        let exp = if n50 < total_hits / 500.0 {
            0.5 * n50
        } else {
            n50 - total_hits / 500.0 * 0.5
        };

        tap_value *= 0.98_f32.powf(exp);

        // Buff very high AR
        let mut ar_factor = 1.0;

        if attributes.ar > 10.33 {
            let ar11_len_buff = 0.8 * (logistic(total_hits / 500.0) - 0.5);
            ar_factor += ar11_len_buff * (attributes.ar - 10.33) / 0.67;
        }

        tap_value *= ar_factor;

        tap_value
    }

    fn compute_acc_value(
        &self,
        attributes: &DifficultyAttributes,
        great_window: f32,
        effective_miss_count: f32,
    ) -> f32 {
        let finger_control_diff = attributes.finger_control_diff;
        let modified_acc = self.modified_acc(attributes);
        let acc_on_circles = modified_acc - 0.003;
        let acc_on_circles_positive = (acc_on_circles - 1.0).exp();

        let deviation_on_circles =
            (great_window + 20.0) / (SQRT_2 * erfinv(acc_on_circles_positive));

        let mut acc_value = deviation_on_circles.powf(-2.2) * finger_control_diff.sqrt() * 46_000.0;

        // scale acc pp with misses
        acc_value *= 0.96_f32.powf((effective_miss_count - MISS_COUNT_LENIENCY).max(0.0));

        // nerf short maps
        let len_factor = if attributes.length < 120.0 {
            logistic((attributes.length - 300.0) / 60.0) + logistic(2.5) - logistic(-2.5)
        } else {
            logistic(attributes.length / 60.0)
        };

        acc_value *= len_factor;

        if self.mods.hd() {
            acc_value *= 1.08;
        }

        if self.mods.fl() {
            acc_value *= 1.02;
        }

        acc_value
    }

    #[inline]
    fn total_hits(&self) -> usize {
        let n_objects = self
            .passed_objects
            .unwrap_or_else(|| self.map.hit_objects.len());

        (self.n300.unwrap_or(0) + self.n100.unwrap_or(0) + self.n50.unwrap_or(0) + self.n_misses)
            .min(n_objects)
    }

    #[inline]
    fn modified_acc(&self, attributes: &DifficultyAttributes) -> f32 {
        let n300 = self.n300.unwrap_or(0);
        let n100 = self.n100.unwrap_or(0);
        let n50 = self.n50.unwrap_or(0);
        let total_hits = self.total_hits();

        ((n300 - (total_hits - attributes.n_circles as usize)) * 3 + n100 * 2 + n50) as f32
            / ((attributes.n_circles + 2) * 3) as f32
    }
}

#[inline]
fn tp_to_pp(tp: f32) -> f32 {
    tp.powf(SKILL_TO_PP_EXPONENT) * 0.118
}

#[inline]
fn tap_skill_to_pp(tap_skill: f32) -> f32 {
    tap_skill.powf(SKILL_TO_PP_EXPONENT) * 0.115
}

pub trait OsuAttributeProvider {
    fn attributes(self) -> DifficultyAttributes;
}

impl OsuAttributeProvider for DifficultyAttributes {
    #[inline]
    fn attributes(self) -> DifficultyAttributes {
        self
    }
}

impl OsuAttributeProvider for PpResult {
    #[inline]
    fn attributes(self) -> DifficultyAttributes {
        self.attributes
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rosu_pp::Beatmap;

    #[test]
    fn osu_only_accuracy() {
        let map = Beatmap::default();

        let total_objects = 1234;
        let target_acc = 97.5;

        let calculator = OsuPP::new(&map)
            .passed_objects(total_objects)
            .accuracy(target_acc);

        let numerator = 6 * calculator.n300.unwrap_or(0)
            + 2 * calculator.n100.unwrap_or(0)
            + calculator.n50.unwrap_or(0);
        let denominator = 6 * total_objects;
        let acc = 100.0 * numerator as f32 / denominator as f32;

        assert!(
            (target_acc - acc).abs() < 1.0,
            "Expected: {} | Actual: {}",
            target_acc,
            acc
        );
    }

    #[test]
    fn osu_accuracy_and_n50() {
        let map = Beatmap::default();

        let total_objects = 1234;
        let target_acc = 97.5;
        let n50 = 30;

        let calculator = OsuPP::new(&map)
            .passed_objects(total_objects)
            .n50(n50)
            .accuracy(target_acc);

        assert!(
            (calculator.n50.unwrap() as i32 - n50 as i32).abs() <= 4,
            "Expected: {} | Actual: {}",
            n50,
            calculator.n50.unwrap()
        );

        let numerator = 6 * calculator.n300.unwrap_or(0)
            + 2 * calculator.n100.unwrap_or(0)
            + calculator.n50.unwrap_or(0);
        let denominator = 6 * total_objects;
        let acc = 100.0 * numerator as f32 / denominator as f32;

        assert!(
            (target_acc - acc).abs() < 1.0,
            "Expected: {} | Actual: {}",
            target_acc,
            acc
        );
    }

    #[test]
    fn osu_missing_objects() {
        let map = Beatmap::default();

        let total_objects = 1234;
        let n300 = 1000;
        let n100 = 200;
        let n50 = 30;

        let mut calculator = OsuPP::new(&map)
            .passed_objects(total_objects)
            .n300(n300)
            .n100(n100)
            .n50(n50);

        calculator.assert_hitresults();

        let n_objects = calculator.n300.unwrap()
            + calculator.n100.unwrap()
            + calculator.n50.unwrap()
            + calculator.n_misses;

        assert_eq!(
            total_objects, n_objects,
            "Expected: {} | Actual: {}",
            total_objects, n_objects
        );
    }
}
