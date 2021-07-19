use super::{
    skill_kind::{
        AIM_COMBINED_STARS_PER_DOUBLE, AIM_FLOW_STARS_PER_DOUBLE, AIM_SNAP_STARS_PER_DOUBLE,
        TAP_STARS_PER_DOUBLE,
    },
    DifficultyObject, SkillKind,
};

use std::cmp::Ordering;
use std::collections::VecDeque;

const STAR_RATING_CONSTANT: f32 = 0.08265;

#[derive(Debug)]
pub(crate) struct Skill<'h> {
    kind: SkillKind,
    previous: VecDeque<DifficultyObject<'h>>,
}

impl<'h> Skill<'h> {
    #[inline]
    pub(crate) fn new(kind: SkillKind) -> Self {
        Self {
            previous: VecDeque::with_capacity(kind.history_len()),
            kind,
        }
    }

    #[inline]
    pub(crate) fn process_internal(&mut self, current: DifficultyObject<'h>) {
        if self.previous.len() > self.kind.history_len() {
            self.previous.pop_back();
        }

        self.process(&current);
        self.previous.push_front(current);
    }

    #[inline]
    fn process(&mut self, current: &DifficultyObject) {
        self.kind.process(current, &self.previous)
    }

    pub(crate) fn calculate_difficulty_value(&self, strains: &[f32], stars_per_double: f32) -> f32 {
        let difficulty_exponent = stars_per_double.log2().recip();
        let mut difficulty = 0.0;

        // println!("diff_exp={}", difficulty_exponent);

        for (_i, &strain) in strains.iter().enumerate() {
            difficulty += strain.powf(difficulty_exponent);
            // println!("{}: {}", _i, strain);
            // println!(
            //     "{}: {} => {}",
            //     _i,
            //     strain.powf(difficulty_exponent),
            //     difficulty
            // );
        }

        difficulty.powf(difficulty_exponent.recip())
    }

    pub(crate) fn combine_star_rating(
        &self,
        first: f32,
        second: f32,
        stars_per_double: f32,
    ) -> f32 {
        let difficulty_exponent = stars_per_double.log2().recip();

        (first.powf(difficulty_exponent) + second.powf(difficulty_exponent))
            .powf(difficulty_exponent.recip())
    }

    pub(crate) fn display_difficulty_value(&mut self) -> f32 {
        match &mut self.kind {
            SkillKind::Aim {
                snap_strains,
                flow_strains,
                ..
            } => {
                let flow_star_rating =
                    calculate_display_difficulty_value(flow_strains, AIM_FLOW_STARS_PER_DOUBLE);
                let snap_star_rating =
                    calculate_display_difficulty_value(snap_strains, AIM_SNAP_STARS_PER_DOUBLE);

                self.combine_star_rating(
                    flow_star_rating,
                    snap_star_rating,
                    AIM_COMBINED_STARS_PER_DOUBLE,
                )
            }
            SkillKind::Tap { strains, .. } => {
                calculate_display_difficulty_value(strains, TAP_STARS_PER_DOUBLE)
            }
        }
    }

    pub(crate) fn difficulty_value(&self) -> f32 {
        match &self.kind {
            SkillKind::Aim {
                snap_strains,
                flow_strains,
                ..
            } => {
                let flow_star_rating =
                    self.calculate_difficulty_value(flow_strains, AIM_FLOW_STARS_PER_DOUBLE);
                let snap_star_rating =
                    self.calculate_difficulty_value(snap_strains, AIM_SNAP_STARS_PER_DOUBLE);

                self.combine_star_rating(
                    flow_star_rating,
                    snap_star_rating,
                    AIM_COMBINED_STARS_PER_DOUBLE,
                )
            }
            SkillKind::Tap { strains, .. } => {
                self.calculate_difficulty_value(strains, TAP_STARS_PER_DOUBLE)
            }
        }
    }
}

pub(crate) fn calculate_display_difficulty_value(
    strains: &mut [f32],
    stars_per_double: f32,
) -> f32 {
    let mut difficulty = 0.0;
    let mut weight = 1.0;
    let decay_weight = 0.95;

    strains.sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

    for &strain in strains.iter() {
        difficulty += strain * weight;
        weight *= decay_weight;
    }

    difficulty * STAR_RATING_CONSTANT
}
