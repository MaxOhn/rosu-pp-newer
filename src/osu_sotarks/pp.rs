use super::Beatmap;

use rosu_pp::{osu::OsuAttributeProvider, Mods, OsuPP as OsuPP_, PpResult};

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
    creator: &'m str,
    mods: u32,
    pp: OsuPP_<'m>,
}

impl<'m> OsuPP<'m> {
    #[inline]
    pub fn new(map: &'m Beatmap) -> Self {
        Self {
            creator: map.creator.as_str(),
            mods: 0,
            pp: OsuPP_::new(&map.inner),
        }
    }

    /// [`OsuAttributeProvider`] is implemented by [`DifficultyAttributes`](crate::osu::DifficultyAttributes)
    /// and by [`PpResult`](crate::PpResult) meaning you can give the
    /// result of a star calculation or a pp calculation.
    /// If you already calculated the attributes for the current map-mod combination,
    /// be sure to put them in here so that they don't have to be recalculated.
    #[inline]
    pub fn attributes(mut self, attributes: impl OsuAttributeProvider) -> Self {
        self.pp = self.pp.attributes(attributes);

        self
    }

    /// Specify mods through their bit values.
    ///
    /// See [https://github.com/ppy/osu-api/wiki#mods](https://github.com/ppy/osu-api/wiki#mods)
    #[inline]
    pub fn mods(mut self, mods: u32) -> Self {
        self.mods = mods;
        self.pp = self.pp.mods(mods);

        self
    }

    /// Specify the max combo of the play.
    #[inline]
    pub fn combo(mut self, combo: usize) -> Self {
        self.pp = self.pp.combo(combo);

        self
    }

    /// Specify the amount of 300s of a play.
    #[inline]
    pub fn n300(mut self, n300: usize) -> Self {
        self.pp = self.pp.n300(n300);

        self
    }

    /// Specify the amount of 100s of a play.
    #[inline]
    pub fn n100(mut self, n100: usize) -> Self {
        self.pp = self.pp.n100(n100);

        self
    }

    /// Specify the amount of 50s of a play.
    #[inline]
    pub fn n50(mut self, n50: usize) -> Self {
        self.pp = self.pp.n50(n50);

        self
    }

    /// Specify the amount of misses of a play.
    #[inline]
    pub fn misses(mut self, n_misses: usize) -> Self {
        self.pp = self.pp.misses(n_misses);

        self
    }

    /// Amount of passed objects for partial plays, e.g. a fail.
    #[inline]
    pub fn passed_objects(mut self, passed_objects: usize) -> Self {
        self.pp = self.pp.passed_objects(passed_objects);

        self
    }

    /// Generate the hit results with respect to the given accuracy between `0` and `100`.
    ///
    /// Be sure to set `misses` beforehand!
    /// In case of a partial play, be also sure to set `passed_objects` beforehand!
    pub fn accuracy(mut self, acc: f32) -> Self {
        self.pp = self.pp.accuracy(acc);

        self
    }

    pub fn calculate(self) -> PpResult {
        let mut result = self.pp.calculate();

        let factor = match self.creator {
            "Sotarks" | "fieryrage" | "Nevo" | "Fatfan Kolek" | "Taeyang" | "Reform"
            | "A r M i N" | "Bibbity Bill" | "Log Off Now" | "Azunyan-" | "DendyHere" => 0.7,
            "Seni" | "Monstrata" | "SnowNiNo_" | "Xexxar" | "Lami" | "Akitoshi" | "Doormat"
            | "Kencho" => 0.85,
            "Nakagawa-Kanon" if self.mods.hd() && self.mods.hr() => 2.0,
            _ => 1.05,
        };

        result.pp *= factor;

        result
    }
}
