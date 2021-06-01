use super::math_util;

use rosu_pp::{
    parse::{
        DifficultyPoint, HitObject, HitObjectKind, ParseError, ParseResult, PathType, Pos2,
        TimingPoint,
    },
    Mods,
};

use std::cmp::Ordering;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

macro_rules! sort {
    ($slice:expr) => {
        $slice.sort_unstable_by(|p1, p2| p1.partial_cmp(&p2).unwrap_or(Ordering::Equal))
    };

    (stable $slice:expr) => {
        $slice.sort_by(|p1, p2| p1.partial_cmp(&p2).unwrap_or(Ordering::Equal))
    };
}

macro_rules! next_field {
    ($opt:expr, $err:literal) => {
        $opt.ok_or_else(|| ParseError::MissingField($err))?
    };
}

macro_rules! validate_float {
    ($x:expr) => {{
        if $x.is_finite() {
            $x
        } else {
            return Err(ParseError::InvalidFloatingPoint);
        }
    }};
}

macro_rules! line_prepare {
    ($buf:ident) => {{
        let mut line = $buf.trim_end();

        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with(' ')
            || line.starts_with('_')
        {
            $buf.clear();
            continue;
        }

        if let Some(idx) = line.find("//") {
            line = &line[..idx];
        }

        line
    }};
}

macro_rules! section {
    ($map:ident, $func:ident, $reader:ident, $buf:ident, $section:ident) => {{
        if $map.$func(&mut $reader, &mut $buf, &mut $section).await? {
            break;
        }
    }};
}

macro_rules! read_line {
    ($reader:ident, $buf:expr) => {{
        $reader.read_line($buf).await
    }};
}

macro_rules! parse_general_body {
    ($self:ident, $reader:ident, $buf:ident, $section:ident) => {{
        let mut mode = None;
        let mut empty = true;
        let mut stack_leniency = None;

        while read_line!($reader, $buf)? != 0 {
            let line = line_prepare!($buf);

            if line.starts_with('[') && line.ends_with(']') {
                *$section = Section::from_str(&line[1..line.len() - 1]);
                empty = false;
                $buf.clear();
                break;
            }

            let (key, value) = split_colon(&line).ok_or(ParseError::BadLine)?;

            if key == "Mode" {
                mode = match value {
                    "0" => Some(GameMode::STD),
                    "1" => Some(GameMode::TKO),
                    "2" => Some(GameMode::CTB),
                    "3" => Some(GameMode::MNA),
                    _ => return Err(ParseError::InvalidMode),
                };
            }

            if key == "StackLeniency" {
                stack_leniency = Some(value.parse()?);
            }

            $buf.clear();
        }

        $self.mode = mode.unwrap_or(GameMode::STD);
        $self.stack_leniency = stack_leniency.unwrap_or(0.7);

        Ok(empty)
    }};
}

macro_rules! parse_general {
    ($reader:ident<$inner:ident>) => {
        fn parse_general<R: $inner>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_general_body!(self, reader, buf, section)
        }
    };

    (async $reader:ident<$inner:ident>) => {
        async fn parse_general<R: $inner + Unpin>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_general_body!(self, reader, buf, section)
        }
    };
}

macro_rules! parse_difficulty_body {
    ($self:ident, $reader:ident, $buf:ident, $section:ident) => {{
        let mut ar = None;
        let mut od = None;
        let mut cs = None;
        let mut hp = None;
        let mut sv = None;
        let mut tick_rate = None;

        let mut empty = true;

        while read_line!($reader, $buf)? != 0 {
            let line = line_prepare!($buf);

            if line.starts_with('[') && line.ends_with(']') {
                *$section = Section::from_str(&line[1..line.len() - 1]);
                empty = false;
                $buf.clear();
                break;
            }

            let (key, value) = split_colon(&line).ok_or(ParseError::BadLine)?;

            match key {
                "ApproachRate" => ar = Some(value.parse()?),
                "OverallDifficulty" => od = Some(value.parse()?),
                "CircleSize" => cs = Some(value.parse()?),
                "HPDrainRate" => hp = Some(value.parse()?),
                "SliderTickRate" => tick_rate = Some(value.parse()?),
                "SliderMultiplier" => sv = Some(value.parse()?),
                _ => {}
            }

            $buf.clear();
        }

        $self.od = next_field!(od, "od");
        $self.cs = next_field!(cs, "cs");
        $self.hp = next_field!(hp, "hp");
        $self.ar = ar.unwrap_or($self.od);
        $self.sv = next_field!(sv, "sv");
        $self.tick_rate = next_field!(tick_rate, "sv");

        Ok(empty)
    }};
}

macro_rules! parse_difficulty {
    ($reader:ident<$inner:ident>) => {
        fn parse_difficulty<R: $inner>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_difficulty_body!(self, reader, buf, section)
        }
    };

    (async $reader:ident<$inner:ident>) => {
        async fn parse_difficulty<R: $inner + Unpin>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_difficulty_body!(self, reader, buf, section)
        }
    };
}

macro_rules! parse_timingpoints_body {
    (short => $self:ident, $reader:ident, $buf:ident, $section:ident) => {{
        let mut empty = true;

        while read_line!($reader, $buf)? != 0 {
            let line = line_prepare!($buf);

            if line.starts_with('[') && line.ends_with(']') {
                *$section = Section::from_str(&line[1..line.len() - 1]);
                empty = false;
                $buf.clear();
                break;
            }

            $buf.clear();
        }

        Ok(empty)
    }};

    ($self:ident, $reader:ident, $buf:ident, $section:ident) => {{
        let mut unsorted_timings = false;
        let mut unsorted_difficulties = false;

        let mut prev_diff = 0.0;
        let mut prev_time = 0.0;

        let mut empty = true;

        while read_line!($reader, $buf)? != 0 {
            let line = line_prepare!($buf);

            if line.starts_with('[') && line.ends_with(']') {
                *$section = Section::from_str(&line[1..line.len() - 1]);
                empty = false;
                $buf.clear();
                break;
            }

            let mut split = line.split(',');

            let time = next_field!(split.next(), "timing point time")
                .trim()
                .parse::<f32>()?;
            validate_float!(time);

            let beat_len = next_field!(split.next(), "beat len")
                .trim()
                .parse::<f32>()?;

            if beat_len < 0.0 {
                let point = DifficultyPoint {
                    time,
                    speed_multiplier: -100.0 / beat_len,
                };

                $self.difficulty_points.push(point);

                if time < prev_diff {
                    unsorted_difficulties = true;
                } else {
                    prev_diff = time;
                }
            } else {
                $self.timing_points.push(TimingPoint { time, beat_len });

                if time < prev_time {
                    unsorted_timings = true;
                } else {
                    prev_time = time;
                }
            }

            $buf.clear();
        }

        if unsorted_timings {
            sort!($self.timing_points);
        }

        if unsorted_difficulties {
            sort!($self.difficulty_points);
        }

        Ok(empty)
    }};
}

macro_rules! parse_timingpoints {
    ($reader:ident<$inner:ident>) => {
        fn parse_timingpoints<R: $inner>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_timingpoints_body!(self, reader, buf, section)
        }
    };

    (async $reader:ident<$inner:ident>) => {
        async fn parse_timingpoints<R: $inner + Unpin>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_timingpoints_body!(self, reader, buf, section)
        }
    };
}

macro_rules! parse_hitobjects_body {
    ($self:ident, $reader:ident, $buf:ident, $section:ident) => {{
        let mut unsorted = false;
        let mut prev_time = 0.0;

        let mut empty = true;

        while read_line!($reader, $buf)? != 0 {
            let line = line_prepare!($buf);

            if line.starts_with('[') && line.ends_with(']') {
                *$section = Section::from_str(&line[1..line.len() - 1]);
                empty = false;
                $buf.clear();
                break;
            }

            let mut split = line.split(',');

            let pos = Pos2 {
                x: next_field!(split.next(), "x position").parse()?,
                y: next_field!(split.next(), "y position").parse()?,
            };

            let time = next_field!(split.next(), "hitobject time")
                .trim()
                .parse::<f32>()?;

            validate_float!(time);

            if !$self.hit_objects.is_empty() && time < prev_time {
                unsorted = true;
            }

            let kind: u8 = next_field!(split.next(), "hitobject kind").parse()?;
            let sound = split.next().map(str::parse).transpose()?.unwrap_or(0);

            let kind = if kind & Self::CIRCLE_FLAG > 0 {
                $self.n_circles += 1;

                HitObjectKind::Circle
            } else if kind & Self::SLIDER_FLAG > 0 {
                $self.n_sliders += 1;

                let mut curve_points = Vec::with_capacity(4);
                curve_points.push(pos);

                let mut curve_point_iter = next_field!(split.next(), "curve points").split('|');

                let mut path_type: PathType =
                    next_field!(curve_point_iter.next(), "path kind").parse()?;

                for pos in curve_point_iter {
                    let mut v = pos.split(':').map(str::parse);

                    match (v.next(), v.next()) {
                        (Some(Ok(x)), Some(Ok(y))) => curve_points.push(Pos2 { x, y }),
                        _ => return Err(ParseError::InvalidCurvePoints),
                    }
                }

                match path_type {
                    PathType::Linear if curve_points.len() % 2 == 0 => {
                        // Assert that the points are of the form A|B|B|C|C|E
                        if math_util::valid_linear(&curve_points) {
                            for i in (2..curve_points.len() - 1).rev().step_by(2) {
                                curve_points.remove(i);
                            }
                        } else {
                            path_type = PathType::Bezier;
                        }
                    }
                    PathType::PerfectCurve if curve_points.len() == 3 => {
                        if math_util::is_linear(curve_points[0], curve_points[1], curve_points[2]) {
                            path_type = PathType::Linear;
                        }
                    }
                    PathType::Catmull => {}
                    _ => path_type = PathType::Bezier,
                };

                // Reduce amount of curvepoints but keep the elements evenly spaced.
                // Necessary to handle maps like XNOR (2573164) which have
                // tens of thousands of curvepoints more efficiently.
                while curve_points.len() > CURVE_POINT_THRESHOLD {
                    let last = curve_points[curve_points.len() - 1];
                    let last_idx = (curve_points.len() - 1) / 2;

                    for i in 1..=last_idx {
                        curve_points.swap(i, 2 * i);
                    }

                    curve_points[last_idx] = last;
                    curve_points.truncate(last_idx + 1);
                }

                if curve_points.is_empty() {
                    HitObjectKind::Circle
                } else {
                    let repeats = next_field!(split.next(), "repeats")
                        .parse::<usize>()?
                        .min(9000);

                    let pixel_len = next_field!(split.next(), "pixel len")
                        .parse::<f32>()?
                        .max(0.0)
                        .min(MAX_COORDINATE_VALUE);

                    HitObjectKind::Slider {
                        repeats,
                        pixel_len,
                        curve_points,
                        path_type,
                    }
                }
            } else if kind & Self::SPINNER_FLAG > 0 {
                $self.n_spinners += 1;
                let end_time = next_field!(split.next(), "spinner endtime").parse()?;

                HitObjectKind::Spinner { end_time }
            } else if kind & Self::HOLD_FLAG > 0 {
                $self.n_sliders += 1;
                let mut end = time;

                if let Some(next) = split.next() {
                    end = end.max(next_field!(next.split(':').next(), "hold endtime").parse()?);
                }

                HitObjectKind::Hold { end_time: end }
            } else {
                return Err(ParseError::UnknownHitObjectKind);
            };

            $self.hit_objects.push(HitObject {
                pos,
                start_time: time,
                kind,
                sound,
            });

            prev_time = time;
            $buf.clear();
        }

        if unsorted {
            sort!($self.hit_objects);
        }

        Ok(empty)
    }};
}

macro_rules! parse_hitobjects {
    ($reader:ident<$inner:ident>) => {
        fn parse_hitobjects<R: $inner>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_hitobjects_body!(self, reader, buf, section)
        }
    };

    (async $reader:ident<$inner:ident>) => {
        async fn parse_hitobjects<R: $inner + Unpin>(
            &mut self,
            reader: &mut $reader<R>,
            buf: &mut String,
            section: &mut Section,
        ) -> ParseResult<bool> {
            parse_hitobjects_body!(self, reader, buf, section)
        }
    };
}

macro_rules! parse_body {
    ($reader:ident<$inner:ident>: $input:ident) => {{
        let mut reader = $reader::new($input);
        let mut buf = String::new();

        loop {
            read_line!(reader, &mut buf)?;

            // Check for character U+FEFF specifically thanks to map id 797130
            if !buf
                .trim_matches(|c: char| c.is_whitespace() || c == '﻿')
                .is_empty()
            {
                break;
            }

            buf.clear();
        }

        let version = match buf.find(OSU_FILE_HEADER) {
            Some(idx) => buf[idx + OSU_FILE_HEADER.len()..].trim_end().parse()?,
            None => return Err(ParseError::IncorrectFileHeader),
        };

        buf.clear();

        let mut map = Beatmap {
            version,
            hit_objects: Vec::with_capacity(256),
            ..Default::default()
        };

        let mut section = Section::None;

        loop {
            match section {
                Section::General => section!(map, parse_general, reader, buf, section),
                Section::Difficulty => section!(map, parse_difficulty, reader, buf, section),
                Section::TimingPoints => section!(map, parse_timingpoints, reader, buf, section),
                Section::HitObjects => section!(map, parse_hitobjects, reader, buf, section),
                Section::None => {
                    if read_line!(reader, &mut buf)? == 0 {
                        break;
                    }

                    let line = line_prepare!(buf);

                    if line.starts_with('[') && line.ends_with(']') {
                        section = Section::from_str(&line[1..line.len() - 1]);
                    }

                    buf.clear();
                }
            }
        }

        Ok(map)
    }};
}

macro_rules! parse {
    ($reader:ident<$inner:ident>) => {
        pub fn parse<R: $inner>(input: R) -> ParseResult<Self> {
            parse_body!($reader<$inner>: input)
        }
    };

    (async $reader:ident<$inner:ident>) => {
        pub async fn parse<R: $inner + Unpin>(input: R) -> ParseResult<Self> {
            parse_body!($reader<$inner>: input)
        }
    };
}

/// The mode of a beatmap.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum GameMode {
    STD = 0,
    TKO = 1,
    CTB = 2,
    MNA = 3,
}

impl Default for GameMode {
    #[inline]
    fn default() -> Self {
        Self::STD
    }
}

/// The main beatmap struct containing all data relevant
/// for difficulty and pp calculation
#[derive(Clone, Default, Debug)]
pub struct Beatmap {
    pub mode: GameMode,
    pub version: u8,

    pub n_circles: u32,
    pub n_sliders: u32,
    pub n_spinners: u32,

    pub ar: f32,
    pub od: f32,
    pub cs: f32,
    pub hp: f32,
    pub sv: f32,
    pub tick_rate: f32,
    pub hit_objects: Vec<HitObject>,
    pub timing_points: Vec<TimingPoint>,
    pub difficulty_points: Vec<DifficultyPoint>,
    pub stack_leniency: f32,
}

pub(crate) const OSU_FILE_HEADER: &str = "osu file format v";

const CURVE_POINT_THRESHOLD: usize = 256;
const MAX_COORDINATE_VALUE: f32 = 131_072.0;

impl Beatmap {
    const CIRCLE_FLAG: u8 = 1 << 0;
    const SLIDER_FLAG: u8 = 1 << 1;
    // const NEW_COMBO_FLAG: u8 = 1 << 2;
    const SPINNER_FLAG: u8 = 1 << 3;
    // const COMBO_OFFSET_FLAG: u8 = (1 << 4) | (1 << 5) | (1 << 6);
    const HOLD_FLAG: u8 = 1 << 7;

    #[inline]
    pub fn attributes(&self) -> BeatmapAttributes {
        BeatmapAttributes::new(self.ar, self.od, self.cs, self.hp)
    }
}

impl Beatmap {
    parse!(async BufReader<AsyncRead>);
    parse_general!(async BufReader<AsyncRead>);
    parse_difficulty!(async BufReader<AsyncRead>);
    parse_timingpoints!(async BufReader<AsyncRead>);
    parse_hitobjects!(async BufReader<AsyncRead>);
}

#[inline]
fn split_colon(line: &str) -> Option<(&str, &str)> {
    let mut split = line.split(':');

    Some((split.next()?, split.next()?.trim()))
}

#[derive(Copy, Clone, Debug)]
enum Section {
    None,
    General,
    Difficulty,
    TimingPoints,
    HitObjects,
}

impl Section {
    #[inline]
    fn from_str(s: &str) -> Self {
        match s {
            "General" => Self::General,
            "Difficulty" => Self::Difficulty,
            "TimingPoints" => Self::TimingPoints,
            "HitObjects" => Self::HitObjects,
            _ => Self::None,
        }
    }
}

/// Summary struct for a [`Beatmap`](crate::Beatmap)'s attributes.
#[derive(Clone, Debug)]
pub struct BeatmapAttributes {
    pub ar: f32,
    pub od: f32,
    pub cs: f32,
    pub hp: f32,
    pub clock_rate: f32,
}

impl BeatmapAttributes {
    const AR0_MS: f32 = 1800.0;
    const AR5_MS: f32 = 1200.0;
    const AR10_MS: f32 = 450.0;
    const AR_MS_STEP_1: f32 = (Self::AR0_MS - Self::AR5_MS) / 5.0;
    const AR_MS_STEP_2: f32 = (Self::AR5_MS - Self::AR10_MS) / 5.0;

    #[inline]
    pub(crate) fn new(ar: f32, od: f32, cs: f32, hp: f32) -> Self {
        Self {
            ar,
            od,
            cs,
            hp,
            clock_rate: 1.0,
        }
    }

    /// Adjusts attributes w.r.t. mods.
    /// AR is further adjusted by its hitwindow.
    /// OD is __not__ adjusted by its hitwindow.
    pub fn mods(self, mods: impl Mods) -> Self {
        if !mods.change_map() {
            return self;
        }

        let clock_rate = mods.speed();
        let multiplier = mods.od_ar_hp_multiplier();

        // AR
        let mut ar = self.ar * multiplier;
        let mut ar_ms = if ar <= 5.0 {
            Self::AR0_MS - Self::AR_MS_STEP_1 * ar
        } else {
            Self::AR5_MS - Self::AR_MS_STEP_2 * (ar - 5.0)
        };

        ar_ms = ar_ms.max(Self::AR10_MS).min(Self::AR0_MS);
        ar_ms /= clock_rate;

        ar = if ar_ms > Self::AR5_MS {
            (Self::AR0_MS - ar_ms) / Self::AR_MS_STEP_1
        } else {
            5.0 + (Self::AR5_MS - ar_ms) / Self::AR_MS_STEP_2
        };

        // OD
        let od = (self.od * multiplier).min(10.0);

        // CS
        let mut cs = self.cs;
        if mods.hr() {
            cs *= 1.3;
        } else if mods.ez() {
            cs *= 0.5;
        }
        cs = cs.min(10.0);

        // HP
        let hp = (self.hp * multiplier).min(10.0);

        Self {
            ar,
            od,
            cs,
            hp,
            clock_rate,
        }
    }
}
