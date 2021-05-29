use super::{DifficultyObject, SkillKind};

use std::cmp::Ordering;
use std::collections::VecDeque;

const TARGET_FC_PRECISION: f32 = 0.1;

#[derive(Debug)]
pub(crate) struct Skill<'h> {
    kind: SkillKind,
    pub(crate) strain_peaks: Vec<f32>,
    pub(crate) times: Vec<f32>,
    previous: VecDeque<DifficultyObject<'h>>,

    curr_strain: f32,
}

impl<'h> Skill<'h> {
    #[inline]
    pub(crate) fn new(kind: SkillKind) -> Self {
        Self {
            kind,
            strain_peaks: Vec::with_capacity(128),
            times: Vec::with_capacity(128),
            previous: VecDeque::with_capacity(kind.history_len()),

            curr_strain: 1.0,
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
        let strain_at = self
            .kind
            .strain_value_at(&mut self.curr_strain, current, &self.previous);

        self.strain_peaks.push(strain_at);
        self.times.push(current.base.time);
    }

    #[inline]
    pub(crate) fn difficulty_value(&mut self) -> f32 {
        let total_difficulty = self.calculate_difficulty_value();

        self.fc_time_skill_level(total_difficulty)
    }

    fn expected_target_time(&self, total_difficulty: f32) -> f32 {
        let mut target_time = 0.0;

        for i in 1..self.strain_peaks.len() {
            target_time += (self.times[i] - self.times[i - 1]).min(2000.0)
                * (self.strain_peaks[i] / total_difficulty);
        }

        target_time
    }

    fn expected_fc_time(&self, skill: f32) -> f32 {
        let mut last_timestamp = self.times[0] - 5.0;
        let mut fc_time = 0.0;

        for i in 0..self.strain_peaks.len() {
            let dt = self.times[i] - last_timestamp;
            last_timestamp = self.times[i];
            let fc_prob = self.fc_probability(skill, self.strain_peaks[i]);
            fc_time = (fc_time + dt) / fc_prob;
        }

        fc_time - (self.times.last().expect("no last") - self.times[0])
    }

    fn fc_time_skill_level(&mut self, total_difficulty: f32) -> f32 {
        let mut length_estimate = 0.4 * (self.times.last().expect("no last") - self.times[0]);
        let target_fc_time = (30 * 60 * 1000) as f32
            + 30.0 * (self.expected_target_time(total_difficulty) - 60_000.0).max(0.0);

        let mut fc_prob = length_estimate / target_fc_time;
        let mut skill = self.skill_level(fc_prob, total_difficulty);

        for _ in 0..5 {
            let fc_time = self.expected_fc_time(skill);
            length_estimate = fc_time * fc_prob;
            fc_prob = length_estimate / target_fc_time;
            skill = self.skill_level(fc_prob, total_difficulty);

            if (fc_time - target_fc_time).abs() < TARGET_FC_PRECISION * target_fc_time {
                break;
            }
        }

        skill
    }

    fn calculate_difficulty_value(&self) -> f32 {
        let difficulty_exponent = self.kind.difficulty_exponent();
        let mut difficulty = 0.0;

        for &strain in self.strain_peaks.iter() {
            difficulty += strain.powf(difficulty_exponent);
        }

        difficulty.powf(difficulty_exponent.recip())
    }

    pub(crate) fn calculate_display_value(&mut self) -> f32 {
        let mut difficulty = 0.0;
        let mut weight = 1.0;
        let decay_weight = 0.9;

        self.strain_peaks
            .sort_unstable_by(|a, b| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

        for strain in self.strain_peaks.iter() {
            difficulty += strain * weight;
            weight *= decay_weight;
        }

        difficulty
    }

    #[inline]
    fn fc_probability(&self, skill: f32, difficulty: f32) -> f32 {
        (-(difficulty / skill.max(1e-10)).powf(self.kind.difficulty_exponent())).exp()
    }

    #[inline]
    fn skill_level(&self, probability: f32, difficulty: f32) -> f32 {
        difficulty * (-probability.ln()).powf(-self.kind.difficulty_exponent().recip())
    }
}
