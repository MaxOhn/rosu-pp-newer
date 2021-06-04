pub struct HermiteSpline {
    x0: f32,
    x1: f32,
    d1: f32,
    val1: f32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
}

impl HermiteSpline {
    #[inline]
    fn new(x0: f32, val0: f32, d0: f32, x1: f32, val1: f32, d1: f32) -> Self {
        let scale = (x1 - x0).recip();
        let scale2 = scale * scale;

        Self {
            x0,
            x1,
            d1,
            val1,
            c0: val0,
            c1: d0,
            c2: 3.0 * (val1 - val0) * scale2 - (2.0 * d0 + d1) * scale,
            c3: (2.0 * (val0 - val1) * scale + d0 + d1) * scale2,
        }
    }

    #[inline]
    fn evaluate(&self, x: f32) -> f32 {
        if x > self.x1 {
            (x - self.x1) * self.d1 + self.val1
        } else if x < self.x0 {
            (x - self.x0) * self.c1 + self.c0
        } else {
            let t = x - self.x0;
            let t2 = t * t;
            let t3 = t2 * t;

            self.c0 + self.c1 * t + self.c2 * t2 + self.c3 * t3
        }
    }
}

pub struct CubicInterpolation {
    splines: Vec<HermiteSpline>,
}

impl CubicInterpolation {
    pub fn new(
        x: &[f32],
        values: &[f32],
        lower_bound_derivative: Option<f32>,
        upper_bound_derivative: Option<f32>,
    ) -> Self {
        let mut derivatives = Vec::with_capacity(x.len());

        derivatives.push(
            lower_bound_derivative
                .unwrap_or_else(|| Self::two_point_derivative(x[0], values[0], x[1], values[1])),
        );

        let iter = (1..x.len() - 1).map(|i| {
            Self::three_point_derivative(
                x[i - 1],
                values[i - 1],
                x[i],
                values[i],
                x[i + 1],
                values[i + 1],
            )
        });

        derivatives.extend(iter);

        let last = x.len() - 1;

        derivatives.push(upper_bound_derivative.unwrap_or_else(|| {
            Self::two_point_derivative(x[last], values[last], x[last - 1], values[last - 1])
        }));

        let splines = (0..x.len() - 1)
            .map(|i| {
                HermiteSpline::new(
                    x[i],
                    values[i],
                    derivatives[i],
                    x[i + 1],
                    values[i + 1],
                    derivatives[i + 1],
                )
            })
            .collect();

        Self { splines }
    }

    #[inline]
    fn two_point_derivative(x: f32, val: f32, x_next: f32, val_next: f32) -> f32 {
        (val_next - val) / (x_next - x)
    }

    #[inline]
    fn three_point_derivative(
        x_prev: f32,
        val_prev: f32,
        x: f32,
        val: f32,
        x_next: f32,
        val_next: f32,
    ) -> f32 {
        ((x_next - x) * Self::two_point_derivative(x_prev, val_prev, x, val)
            + (x - x_prev) * Self::two_point_derivative(x, val, x_next, val_next))
            / (x_next - x_prev)
    }

    #[inline]
    pub fn evaluate(&self, x: f32) -> f32 {
        let idx = self.spline_index(x);

        self.evaluate_idx(idx, x)
    }

    #[inline]
    fn evaluate_idx(&self, idx: usize, x: f32) -> f32 {
        self.splines[idx].evaluate(x)
    }

    #[inline]
    fn spline_index(&self, x: f32) -> usize {
        let mut i = self.splines.len() - 1;

        while i > 0 && self.splines[i].x0 > x {
            i -= 1;
        }

        i
    }
}

struct BicubicInterpolation {
    x_arr: &'static [f32],
    dx_lower: Option<f32>,
    dx_upper: Option<f32>,
    cubic_interpolations: Vec<CubicInterpolation>,
}

impl BicubicInterpolation {
    fn new<U>(
        x: &'static [f32],
        y: &[f32],
        values: U,
        dx_lower: Option<f32>,
        dx_upper: Option<f32>,
        dy_lower: Option<f32>,
        dy_upper: Option<f32>,
    ) -> Self
    where
        U: IntoIterator,
        U::Item: AsRef<[f32]>,
    {
        let mut cubic_interpolations = Vec::with_capacity(x.len());

        let iter = values
            .into_iter()
            .map(|value| CubicInterpolation::new(y, value.as_ref(), dy_lower, dy_upper));

        cubic_interpolations.extend(iter);

        Self {
            x_arr: x,
            dx_lower,
            dx_upper,
            cubic_interpolations,
        }
    }

    #[inline]
    fn spline_index(&self, x: f32, y: f32) -> (usize, usize) {
        let x_idx = self
            .x_arr
            .iter()
            .rev()
            .skip(1)
            .position(|&elem| elem <= x)
            .map(|i| self.x_arr.len() - i - 2)
            .unwrap_or(0);

        (x_idx, self.cubic_interpolations[0].spline_index(y))
    }

    fn evaluate_idx(&self, x_idx: usize, y_idx: usize, x: f32, y: f32) -> f32 {
        let x0 = self.x_arr[x_idx];
        let x1 = self.x_arr[x_idx + 1];

        let val0 = self.cubic_interpolations[x_idx].evaluate_idx(y_idx, y);
        let val1 = self.cubic_interpolations[x_idx + 1].evaluate_idx(y_idx, y);

        let d0 = if x_idx == 0 {
            self.dx_lower
                .unwrap_or_else(|| CubicInterpolation::two_point_derivative(x0, val0, x1, val1))
        } else {
            let x_prev = self.x_arr[x_idx - 1];
            let val_prev = self.cubic_interpolations[x_idx - 1].evaluate_idx(y_idx, y);

            CubicInterpolation::three_point_derivative(x_prev, val_prev, x0, val0, x1, val1)
        };

        let d1 = if x_idx == self.cubic_interpolations.len() - 2 {
            self.dx_upper
                .unwrap_or_else(|| CubicInterpolation::two_point_derivative(x0, val0, x1, val1))
        } else {
            let x2 = self.x_arr[x_idx + 2];
            let val2 = self.cubic_interpolations[x_idx + 2].evaluate_idx(y_idx, y);

            CubicInterpolation::three_point_derivative(x0, val0, x1, val1, x2, val2)
        };

        HermiteSpline::new(x0, val0, d0, x1, val1, d1).evaluate(x)
    }
}

pub struct TricubicInterpolation {
    x_arr: &'static [f32],
    dx_lower: Option<f32>,
    dx_upper: Option<f32>,
    cubic_interpolations: Vec<BicubicInterpolation>,
}

impl TricubicInterpolation {
    pub fn new<U, V>(
        x: &'static [f32],
        y: &'static [f32],
        z: &[f32],
        values: V,
        mut dx_lower: Option<f32>,
        mut dx_upper: Option<f32>,
        mut dy_lower: Option<f32>,
        mut dy_upper: Option<f32>,
        mut dz_lower: Option<f32>,
        mut dz_upper: Option<f32>,
    ) -> Self
    where
        U: IntoIterator,
        U::Item: AsRef<[f32]>,
        V: IntoIterator<Item = U>,
    {
        dx_lower.get_or_insert(0.0);
        dx_upper.get_or_insert(0.0);
        dy_lower.get_or_insert(0.0);
        dy_upper.get_or_insert(0.0);
        dz_lower.get_or_insert(0.0);
        dz_upper.get_or_insert(0.0);

        let mut cubic_interpolations = Vec::with_capacity(x.len());

        let iter = values.into_iter().map(|value| {
            BicubicInterpolation::new(y, z, value, dy_lower, dy_upper, dz_lower, dz_upper)
        });

        cubic_interpolations.extend(iter);

        Self {
            x_arr: x,
            dx_lower,
            dx_upper,
            cubic_interpolations,
        }
    }

    #[inline]
    fn spline_index(&self, x: f32) -> usize {
        let i = self
            .x_arr
            .iter()
            .rev()
            .skip(1)
            .position(|&elem| elem <= x)
            .map(|i| self.x_arr.len() - i - 2)
            .unwrap_or(0);

        i
    }

    pub fn evaluate(&self, x: f32, y: f32, z: f32) -> f32 {
        let x_idx = self.spline_index(x);
        let (y_idx, z_idx) = self.cubic_interpolations[0].spline_index(y, z);

        let x0 = self.x_arr[x_idx];
        let x1 = self.x_arr[x_idx + 1];

        let val0 = self.cubic_interpolations[x_idx].evaluate_idx(y_idx, z_idx, y, z);
        let val1 = self.cubic_interpolations[x_idx + 1].evaluate_idx(y_idx, z_idx, y, z);

        let d0 = if x_idx == 0 {
            self.dx_lower
                .unwrap_or_else(|| CubicInterpolation::two_point_derivative(x0, val0, x1, val1))
        } else {
            let x_prev = self.x_arr[x_idx - 1];
            let val_prev = self.cubic_interpolations[x_idx - 1].evaluate_idx(y_idx, z_idx, y, z);

            CubicInterpolation::three_point_derivative(x_prev, val_prev, x0, val0, x1, val1)
        };

        let d1 = if x_idx == self.cubic_interpolations.len() - 2 {
            self.dx_upper
                .unwrap_or_else(|| CubicInterpolation::two_point_derivative(x0, val0, x1, val1))
        } else {
            let x2 = self.x_arr[x_idx + 2];
            let val2 = self.cubic_interpolations[x_idx + 2].evaluate_idx(y_idx, z_idx, y, z);

            CubicInterpolation::three_point_derivative(x0, val0, x1, val1, x2, val2)
        };

        HermiteSpline::new(x0, val0, d0, x1, val1, d1).evaluate(x)
    }
}
