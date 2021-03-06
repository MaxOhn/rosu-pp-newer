use super::{CubicInterpolation, TricubicInterpolation};

pub const FLOW_NEG2_VALUES: [[[f32; 5]; 6]; 6] = [
    [
        // d1=0.2
        // 0,   45,   90,   135,   180 degrees
        [0.45, 0.44, 0.42, 0.39, 0.39], // d2=0.1
        [0.89, 0.87, 0.80, 0.72, 0.67], // d2=0.6
        [0.99, 0.99, 0.98, 0.97, 0.96], // d2=1
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=1.3
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=1.8
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=3
    ],
    [
        // d1=0.6
        // 0,   45,   90,   135,   180 degrees
        [0.27, 0.26, 0.23, 0.20, 0.19], // d2=0.1
        [0.75, 0.68, 0.44, 0.26, 0.20], // d2=0.6
        [0.97, 0.94, 0.83, 0.59, 0.46], // d2=1
        [0.99, 0.99, 0.96, 0.86, 0.77], // d2=1.3
        [1.00, 1.00, 1.00, 0.99, 0.98], // d2=1.8
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=3
    ],
    [
        // d1=1
        // 0,   45,   90,   135,   180 degrees
        [0.16, 0.16, 0.14, 0.13, 0.13], // d2=0.1
        [0.37, 0.31, 0.19, 0.13, 0.11], // d2=0.6
        [0.65, 0.55, 0.29, 0.20, 0.17], // d2=1
        [0.83, 0.76, 0.51, 0.34, 0.28], // d2=1.3
        [0.96, 0.94, 0.84, 0.69, 0.61], // d2=1.8
        [1.00, 1.00, 1.00, 0.99, 0.99], // d2=3
    ],
    [
        // d1=1.3
        // 0,   45,   90,   135,   180 degrees
        [0.29, 0.28, 0.26, 0.23, 0.23], // d2=0.1
        [0.56, 0.48, 0.31, 0.19, 0.16], // d2=0.6
        [0.80, 0.71, 0.41, 0.24, 0.18], // d2=1
        [0.91, 0.85, 0.61, 0.34, 0.25], // d2=1.3
        [0.98, 0.96, 0.87, 0.63, 0.49], // d2=1.8
        [1.00, 1.00, 1.00, 0.99, 0.97], // d2=3
    ],
    [
        // d1=1.7
        // 0,   45,   90,   135,   180 degrees
        [0.39, 0.38, 0.35, 0.32, 0.31], // d2=0.1
        [0.66, 0.59, 0.39, 0.24, 0.20], // d2=0.6
        [0.85, 0.78, 0.47, 0.24, 0.17], // d2=1
        [0.93, 0.88, 0.62, 0.27, 0.18], // d2=1.3
        [0.98, 0.97, 0.84, 0.45, 0.27], // d2=1.8
        [1.00, 1.00, 0.99, 0.94, 0.85], // d2=3
    ],
    [
        // d1=2.1
        // 0,   45,   90,   135,   180 degrees
        [0.94, 0.94, 0.93, 0.92, 0.92], // d2=0.1
        [0.98, 0.97, 0.94, 0.89, 0.86], // d2=0.6
        [0.99, 0.99, 0.96, 0.87, 0.82], // d2=1
        [1.00, 0.99, 0.97, 0.87, 0.81], // d2=1.3
        [1.00, 1.00, 0.99, 0.90, 0.84], // d2=1.8
        [1.00, 1.00, 1.00, 0.99, 0.98], // d2=3
    ],
];

pub const FLOW_NEXT_VALUES: [[[f32; 5]; 6]; 5] = [
    [
        // d1=0.2
        // 0,   45,   90,   135,   180 degrees
        [0.02, 0.02, 0.02, 0.03, 0.03], // d2=0.1
        [0.07, 0.08, 0.11, 0.13, 0.14], // d2=0.6
        [0.24, 0.27, 0.32, 0.37, 0.39], // d2=1
        [0.47, 0.50, 0.57, 0.62, 0.64], // d2=1.3
        [0.84, 0.85, 0.88, 0.90, 0.91], // d2=1.8
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=3
    ],
    [
        // d1=0.6
        // 0,   45,   90,   135,   180 degrees
        [0.03, 0.03, 0.03, 0.03, 0.04], // d2=0.1
        [0.04, 0.06, 0.09, 0.13, 0.15], // d2=0.6
        [0.09, 0.15, 0.25, 0.36, 0.41], // d2=1
        [0.22, 0.30, 0.46, 0.60, 0.65], // d2=1.3
        [0.60, 0.68, 0.81, 0.89, 0.91], // d2=1.8
        [0.99, 0.99, 1.00, 1.00, 1.00], // d2=3
    ],
    [
        // d1=1
        // 0,   45,   90,   135,   180 degrees
        [0.04, 0.04, 0.04, 0.05, 0.05], // d2=0.1
        [0.05, 0.06, 0.10, 0.15, 0.18], // d2=0.6
        [0.07, 0.12, 0.21, 0.37, 0.43], // d2=1
        [0.11, 0.21, 0.38, 0.59, 0.66], // d2=1.3
        [0.36, 0.52, 0.74, 0.87, 0.91], // d2=1.8
        [0.96, 0.98, 0.99, 1.00, 1.00], // d2=3
    ],
    [
        // d1=1.5
        // 0,   45,   90,   135,   180 degrees
        [0.07, 0.07, 0.07, 0.08, 0.08], // d2=0.1
        [0.12, 0.14, 0.19, 0.26, 0.29], // d2=0.6
        [0.20, 0.25, 0.38, 0.53, 0.59], // d2=1
        [0.29, 0.40, 0.56, 0.74, 0.79], // d2=1.3
        [0.56, 0.71, 0.84, 0.93, 0.95], // d2=1.8
        [0.98, 0.99, 1.00, 1.00, 1.00], // d2=3
    ],
    [
        // d1=2.8
        // 0,   45,   90,   135,   180 degrees
        [0.10, 0.10, 0.10, 0.10, 0.11], // d2=0.1
        [0.36, 0.37, 0.38, 0.39, 0.40], // d2=0.6
        [0.67, 0.68, 0.70, 0.72, 0.72], // d2=1
        [0.85, 0.85, 0.86, 0.88, 0.88], // d2=1.3
        [0.97, 0.97, 0.97, 0.98, 0.98], // d2=1.8
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=3
    ],
];

pub const SNAP_NEG2_VALUES: [[[f32; 5]; 5]; 7] = [
    [
        // d1=0.6
        // 0,   45,   90,   135,   180 degrees
        [0.52, 0.52, 0.52, 0.52, 0.52], // d2=0
        [0.34, 0.40, 0.56, 0.72, 0.77], // d2=1
        [0.43, 0.52, 0.74, 0.88, 0.91], // d2=2
        [0.68, 0.76, 0.89, 0.95, 0.97], // d2=3
        [0.95, 0.97, 0.98, 0.98, 0.99], // d2=5
    ],
    [
        // d1=1.5
        // 0,   45,   90,   135,   180 degrees
        [0.76, 0.76, 0.76, 0.76, 0.76], // d2=0
        [0.37, 0.48, 0.65, 0.48, 0.94], // d2=1
        [0.21, 0.36, 0.73, 0.55, 0.98], // d2=2
        [0.32, 0.52, 0.92, 0.99, 1.00], // d2=3
        [0.90, 0.96, 1.00, 1.00, 1.00], // d2=5
    ],
    [
        // d1=2.4
        // 0,   45,   90,   135,   180 degrees
        [0.45, 0.45, 0.45, 0.45, 0.45], // d2=0
        [0.12, 0.18, 0.35, 0.61, 0.81], // d2=1.2
        [0.05, 0.11, 0.42, 0.73, 0.96], // d2=2.4
        [0.07, 0.17, 0.60, 0.98, 1.00], // d2=3.6
        [0.56, 0.77, 0.99, 1.00, 1.00], // d2=6
    ],
    [
        // d1=3.5
        // 0,   45,   90,   135,   180 degrees
        [0.37, 0.37, 0.37, 0.37, 0.37], // d2=0
        [0.07, 0.12, 0.38, 0.76, 0.88], // d2=1.75
        [0.02, 0.08, 0.51, 0.96, 1.00], // d2=3.5
        [0.03, 0.16, 0.81, 1.00, 1.00], // d2=5.25
        [0.57, 0.87, 1.00, 1.00, 1.00], // d2=8.75
    ],
    [
        // d1=5
        // 0,   45,   90,   135,   180 degrees
        [0.27, 0.27, 0.27, 0.27, 0.27], // d2=0
        [0.08, 0.13, 0.31, 0.58, 0.69], // d2=2.5
        [0.04, 0.14, 0.48, 0.84, 0.90], // d2=5
        [0.16, 0.33, 0.78, 0.94, 0.96], // d2=7.5
        [0.85, 0.92, 0.96, 0.97, 0.97], // d2=12.5
    ],
    [
        // d1=6.5
        // 0,   45,   90,   135,   180 degrees
        [0.26, 0.26, 0.26, 0.26, 0.26], // d2=0
        [0.13, 0.16, 0.27, 0.44, 0.53], // d2=2.5
        [0.08, 0.15, 0.32, 0.65, 0.77], // d2=5
        [0.17, 0.24, 0.49, 0.83, 0.90], // d2=7.5
        [0.62, 0.71, 0.90, 0.98, 0.99], // d2=12.5
    ],
    [
        // d1=9
        // 0,   45,   90,   135,   180 degrees
        [0.26, 0.26, 0.26, 0.26, 0.26], // d2=0
        [0.13, 0.16, 0.27, 0.44, 0.53], // d2=2.5
        [0.08, 0.15, 0.32, 0.65, 0.77], // d2=5
        [0.17, 0.24, 0.49, 0.83, 0.90], // d2=7.5
        [0.62, 0.71, 0.90, 0.98, 0.99], // d2=12.5
    ],
];

pub const SNAP_NEXT_VALUES: [[[f32; 5]; 5]; 7] = [
    [
        // d1=0.6
        // 0,   45,   90,   135,   180 degrees
        [0.62, 0.62, 0.62, 0.62, 0.62], // d2=0
        [0.80, 0.77, 0.66, 0.54, 0.49], // d2=1
        [0.92, 0.89, 0.78, 0.59, 0.50], // d2=2
        [0.97, 0.96, 0.90, 0.76, 0.66], // d2=3
        [1.00, 1.00, 0.99, 0.96, 0.94], // d2=5
    ],
    [
        // d1=1.5
        // 0,   45,   90,   135,   180 degrees
        [0.62, 0.62, 0.62, 0.62, 0.62], // d2=0
        [0.76, 0.72, 0.66, 0.54, 0.49], // d2=1
        [0.88, 0.82, 0.78, 0.59, 0.50], // d2=2
        [0.97, 0.96, 0.90, 0.76, 0.66], // d2=3
        [1.00, 1.00, 0.99, 0.96, 0.94], // d2=5
    ],
    [
        // d1=2.4
        // 0,   45,   90,   135,   180 degrees
        [0.12, 0.12, 0.12, 0.12, 0.12], // d2=0
        [0.50, 0.35, 0.27, 0.16, 0.13], // d2=1.2
        [0.80, 0.62, 0.49, 0.24, 0.17], // d2=2.4
        [0.95, 0.91, 0.74, 0.43, 0.31], // d2=3.6
        [1.00, 0.99, 0.97, 0.88, 0.80], // d2=6
    ],
    [
        // d1=3.5
        // 0,   45,   90,   135,   180 degrees
        [0.08, 0.08, 0.08, 0.08, 0.08], // d2=0
        [0.68, 0.53, 0.25, 0.09, 0.05], // d2=1.75
        [0.94, 0.88, 0.64, 0.18, 0.08], // d2=3.5
        [1.00, 0.99, 0.93, 0.57, 0.31], // d2=5.25
        [1.00, 1.00, 1.00, 0.99, 0.97], // d2=8.75
    ],
    [
        // d1=5
        // 0,   45,   90,   135,   180 degrees
        [0.11, 0.11, 0.11, 0.11, 0.11], // d2=0
        [0.88, 0.77, 0.39, 0.10, 0.05], // d2=2.5
        [0.99, 0.99, 0.86, 0.29, 0.07], // d2=5
        [1.00, 1.00, 0.99, 0.83, 0.53], // d2=7.5
        [1.00, 1.00, 1.00, 1.00, 1.00], // d2=12.5
    ],
    [
        // d1=6.5
        // 0,   45,   90,   135,   180 degrees
        [0.09, 0.09, 0.09, 0.09, 0.09], // d2=0
        [0.79, 0.66, 0.32, 0.10, 0.06], // d2=2.5
        [0.98, 0.96, 0.76, 0.22, 0.07], // d2=5
        [1.00, 1.00, 0.97, 0.66, 0.29], // d2=7.5
        [1.00, 1.00, 1.00, 0.99, 0.98], // d2=12.5
    ],
    [
        // d1=9
        // 0,   45,   90,   135,   180 degrees
        [0.09, 0.09, 0.09, 0.09, 0.09], // d2=0
        [0.79, 0.66, 0.32, 0.10, 0.06], // d2=2.5
        [0.98, 0.96, 0.76, 0.22, 0.07], // d2=5
        [1.00, 1.00, 0.97, 0.66, 0.29], // d2=7.5
        [1.00, 1.00, 1.00, 0.99, 0.98], // d2=12.5
    ],
];

const ANGLE: f32 = std::f32::consts::FRAC_PI_4;
const ANGLES: [f32; 5] = [0.0, ANGLE, 2.0 * ANGLE, 3.0 * ANGLE, 4.0 * ANGLE];

pub struct AngleCorrection {
    interpolation: TricubicInterpolation,
    min: Option<CubicInterpolation>,
    max: Option<CubicInterpolation>,
    d2_scale: Box<dyn Fn(f32) -> f32 + Send + Sync>,
}

lazy_static::lazy_static! {
    pub static ref FLOW_NEG2: AngleCorrection = AngleCorrection::new(
        &[0.2, 0.6, 1.0, 1.3, 1.7, 2.1],
        &[0.1, 0.6, 1.0, 1.3, 1.8, 3.0],
        &ANGLES,
        &FLOW_NEG2_VALUES,
        None,
        None,
        Box::new(|_| 1.0),
    );

    pub static ref FLOW_NEXT: AngleCorrection = AngleCorrection::new(
        &[0.2, 0.6, 1.0, 1.5, 2.8],
        &[0.1, 0.6, 1.0, 1.3, 1.8, 3.0],
        &ANGLES,
        &FLOW_NEXT_VALUES,
        None,
        None,
        Box::new(|_| 1.0),
    );

    pub static ref SNAP_NEG2: AngleCorrection = AngleCorrection::new(
        &[0.6, 1.5, 2.4, 3.5, 5.0, 6.5, 9.0],
        &[0.0, 0.5, 1.0, 1.5, 2.5],
        &ANGLES,
        &SNAP_NEG2_VALUES,
        None,
        Some(CubicInterpolation::new(
            &[0.0, 1.5, 2.5, 4.0, 6.0, 6.01],
            &[1.0, 0.85, 0.6, 0.8, 1.0, 1.0],
            None,
            None,
        )),
        Box::new(|v: f32| v.clamp(2.0, 5.0)),
    );

    pub static ref SNAP_NEXT: AngleCorrection = AngleCorrection::new(
        &[0.6, 1.5, 2.4, 3.5, 5.0, 6.5, 9.0],
        &[0.0, 0.5, 1.0, 1.5, 2.5],
        &ANGLES,
        &SNAP_NEXT_VALUES,
        None,
        None,
        Box::new(|v: f32| v.clamp(2.0, 5.0)),
    );
}

impl AngleCorrection {
    fn new<F, U, V>(
        d1: &'static [f32],
        d2: &'static [f32],
        angles: &[f32],
        values: V,
        min: Option<CubicInterpolation>,
        max: Option<CubicInterpolation>,
        d2_scale: Box<F>,
    ) -> Self
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
        U: IntoIterator,
        U::Item: AsRef<[f32]>,
        V: IntoIterator<Item = U>,
    {
        let interpolation =
            TricubicInterpolation::new(d1, d2, angles, values, None, None, None, None, None, None);

        Self {
            interpolation,
            min,
            max,
            d2_scale,
        }
    }

    pub fn evaluate(&self, dist1: f32, mut x: f32, mut y: f32) -> f32 {
        let dist2scale = (self.d2_scale)(dist1).recip();

        x *= dist2scale;
        y *= dist2scale;

        let angle = y.atan2(x).abs();
        let dist2 = x.hypot(y);
        let max_val = self.max.as_ref().map_or(1.0, |max| max.evaluate(dist1));
        let min_val = self.min.as_ref().map_or(0.0, |min| min.evaluate(dist1));
        let scale = max_val - min_val;

        let interpolation = self.interpolation.evaluate(dist1, dist2, angle);

        min_val + scale * interpolation.clamp(0.0, 1.0)
    }
}
