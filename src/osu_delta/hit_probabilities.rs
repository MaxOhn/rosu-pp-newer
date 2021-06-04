use super::{math_util, Movement};

use std::collections::HashMap;
use std::f32::consts::SQRT_2;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone)]
struct SkillData {
    expected_time: f32,
    fc_prob: f32,
}

impl Default for SkillData {
    #[inline]
    fn default() -> Self {
        Self {
            expected_time: 0.0,
            fc_prob: 1.0,
        }
    }
}

struct MapSectionCache<'m> {
    cache: HashMap<F32, SkillData>,
    cheese_level: f32,
    start_t: f32,
    end_t: f32,
    movements: &'m [Movement],
}

#[derive(Copy, Clone)]
struct F32(f32);

impl PartialEq for F32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 as u32 == other.0 as u32
    }
}

impl Eq for F32 {}

impl Hash for F32 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.0 as u32);
        state.finish();
    }
}

impl<'m> MapSectionCache<'m> {
    #[inline]
    fn new(movements: &'m [Movement], cheese_level: f32, start_t: f32, end_t: f32) -> Self {
        Self {
            movements,
            start_t,
            end_t,
            cheese_level,
            cache: HashMap::new(),
        }
    }

    fn evaluate(&mut self, tp: f32) -> SkillData {
        let mut result = SkillData::default();

        if self.movements.is_empty() {
            return result;
        }

        let key = F32(tp);

        if let Some(result) = self.cache.get(&key) {
            return *result;
        }

        for movement in self.movements {
            let mut hit_prob =
                HitProbabilities::calculate_cheese_hit_prob(movement, tp, self.cheese_level)
                    + 1e-10;
            hit_prob = 1.0 - ((1.0 - hit_prob + 0.25).sqrt() - 0.5);

            result.expected_time = (result.expected_time + movement.raw_movement_time) / hit_prob;
            result.fc_prob *= hit_prob;
        }

        self.cache.insert(key, result);

        result
    }
}

pub(crate) struct HitProbabilities<'m> {
    sections: Vec<MapSectionCache<'m>>,
}

impl<'m> HitProbabilities<'m> {
    pub(crate) fn new(
        movements: &'m [Movement],
        cheese_level: f32,
        difficulty_count: usize,
    ) -> Self {
        let mut sections = Vec::with_capacity(difficulty_count);

        let iter = (0..difficulty_count).map(|i| {
            let start = movements.len() * i / difficulty_count;
            let end = movements.len() * (i + 1) / difficulty_count - 1;
            let start_t = movements[start].time;
            let end_t = movements[end].time;

            MapSectionCache::new(&movements[start..end + 1], cheese_level, start_t, end_t)
        });

        sections.extend(iter);

        Self { sections }
    }

    pub(crate) fn min_expected_time_for_section_count(
        &mut self,
        tp: f32,
        section_count: usize,
    ) -> f32 {
        let mut fc_time = f32::INFINITY;
        let section_data: Vec<_> = self.sections.iter_mut().map(|s| s.evaluate(tp)).collect();

        for i in 0..=self.sections.len() - section_count {
            fc_time = fc_time.min(
                expected_fc_time(&section_data, i, section_count) - self.length(i, section_count),
            );
        }

        fc_time
    }

    #[inline]
    fn length(&self, start: usize, section_count: usize) -> f32 {
        self.sections[start + section_count - 1].end_t - self.sections[start].start_t
    }

    pub(crate) fn calculate_cheese_hit_prob(
        movement: &Movement,
        tp: f32,
        cheese_level: f32,
    ) -> f32 {
        let mut per_movement_cheese_level = cheese_level;

        if movement.ends_on_slider {
            per_movement_cheese_level = 0.5 * cheese_level + 0.5;
        }

        let cheese_mt =
            movement.movement_time * (1.0 + per_movement_cheese_level * movement.cheesable_ratio);

        calculate_hit_prob(movement.dist, cheese_mt, tp)
    }
}

#[inline]
fn expected_fc_time(section_data: &[SkillData], start: usize, count: usize) -> f32 {
    let mut fc_time = 15.0;

    for skill in section_data.iter().skip(start).take(count) {
        fc_time /= skill.fc_prob;
        fc_time += skill.expected_time;
    }

    fc_time
}

#[inline]
fn calculate_hit_prob(d: f32, mt: f32, tp: f32) -> f32 {
    if d.abs() < f32::EPSILON || mt * tp > 50.0 {
        return 1.0;
    }

    math_util::erf(2.066 / d * ((mt.max(0.03) * tp).exp2() - 1.0) / SQRT_2)
}
