use std::f32::consts::{PI, SQRT_2};

#[inline]
pub(crate) fn logistic(x: f32) -> f32 {
    let ex = x.exp();

    (ex / (ex + 1.0)).min(1.0)
}

#[inline]
pub(crate) fn fitts_ip(d: f32, mt: f32) -> f32 {
    (d + 1.0).log2() / (mt + 1e-10)
}

#[inline]
pub(crate) fn erf(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 - erfccheb(x)
    } else {
        erfccheb(-x) - 1.0
    }
}

#[inline]
pub(crate) fn erfc(x: f32) -> f32 {
    if x >= 0.0 {
        erfccheb(x)
    } else {
        2.0 - erfccheb(-x)
    }
}

fn erfccheb(z: f32) -> f32 {
    let mut d = 0.0;
    let mut dd = 0.0;

    debug_assert!(z >= 0.0, "erfccheb requires nonnegative argument");

    let t = 2.0 / (2.0 + z);
    let ty = 4.0 * t - 2.0;

    for j in (1..NCOEF - 1).rev() {
        let tmp = d;
        d = ty * d - dd + COF[j];
        dd = tmp;
    }

    t * (-z.powi(2) + 0.5 * (COF[0] + ty * d) - dd).exp()
}

pub(crate) fn erfcinv(p: f32) -> f32 {
    if p >= 2.0 {
        return -100.0;
    } else if p <= 0.0 {
        return 100.0;
    }

    let pp = if p < 1.0 { p } else { 2.0 - p };
    let t = (-2.0 * (pp / 2.0).ln()).sqrt();
    let mut x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);

    for _ in 0..2 {
        let err = erfc(x) - pp;
        x += err / (1.12837916709551257 * (-x.powi(2)).exp() - x * err);
    }

    if p < 1.0 {
        x
    } else {
        -x
    }
}

#[inline]
pub(crate) fn erfinv(p: f32) -> f32 {
    erfcinv(1.0 - p)
}

const NCOEF: usize = 28;
const COF: [f32; 28] = [
    -1.3026537197817094,
    6.4196979235649026e-1,
    1.9476473204185836e-2,
    -9.561514786808631e-3,
    -9.46595344482036e-4,
    3.66839497852761e-4,
    4.2523324806907e-5,
    -2.0278578112534e-5,
    -1.624290004647e-6,
    1.303655835580e-6,
    1.5626441722e-8,
    -8.5238095915e-8,
    6.529054439e-9,
    5.059343495e-9,
    -9.91364156e-10,
    -2.27365122e-10,
    9.6467911e-11,
    2.394038e-12,
    -6.886027e-12,
    8.94487e-13,
    3.13092e-13,
    -1.12708e-13,
    3.81e-16,
    7.106e-15,
    -1.523e-15,
    -9.4e-17,
    1.21e-16,
    -2.8e-17,
];

pub(crate) fn try_find_root_brent(
    mut f: impl FnMut(f32) -> f32,
    mut lower: f32,
    mut upper: f32,
    acc: f32,
    iters: usize,
) -> Option<f32> {
    let mut f_min = f(lower);
    let mut f_max = f(upper);
    let mut f_root = f_max;
    let mut d = 0.0;
    let mut e = 0.0;

    if f_min.signum() == f_max.signum() {
        eprintln!("root must be bracketed");

        return None;
    }

    let mut root = upper;
    let mut x_mid = f32::NAN;

    for _ in 0..=iters {
        if f_root.signum() == f_max.signum() {
            upper = lower;
            f_max = f_min;
            e = root - lower;
            d = e;
        }

        if f_max.abs() < f_root.abs() {
            lower = root;
            root = upper;
            upper = lower;
            f_min = f_root;
            f_root = f_max;
            f_max = f_min;
        }

        // ! Required to increase this to make it work
        // let x_acc1 = f32::EPSILON * root.abs() + 0.5 * acc;
        let x_acc1 = 10_000.0 * f32::EPSILON * root.abs() + 0.5 * acc;

        let x_mid_old = x_mid;
        x_mid = (upper - root) / 2.0;

        if x_mid.abs() <= x_acc1 || almost_equal_norm_relative(f_root, 0.0, f_root, acc) {
            return Some(root);
        }

        if (x_mid - x_mid_old).abs() < f32::EPSILON {
            eprintln!("accuracy not sufficient, but cannot be improved further");

            return None;
        }

        if e.abs() >= x_acc1 && f_min.abs() > f_root.abs() {
            let s = f_root / f_min;
            let mut p;
            let mut q;

            if almost_equal_relative(lower, upper) {
                p = 2.0 * x_mid * s;
                q = 1.0 - s;
            } else {
                q = f_min / f_max;
                let r = f_root / f_max;
                p = s * (2.0 * x_mid * q * (q - r) - (root - lower) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }

            if p > 0.0 {
                q = -q;
            }

            p = p.abs();

            if 2.0 * p < (3.0 * x_mid * q - (x_acc1 * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = x_mid;
                e = d;
            }
        } else {
            d = x_mid;
            e = d;
        }

        lower = root;
        f_min = f_root;

        if d.abs() > x_acc1 {
            root += d;
        } else {
            root += x_acc1 * x_mid.signum();
        }

        f_root = f(root);
    }

    eprintln!("no root after max iterations");

    None
}

pub(crate) fn try_find_root_bisection(
    mut f: impl FnMut(f32) -> f32,
    mut lower: f32,
    mut upper: f32,
    acc: f32,
    iters: usize,
) -> Option<f32> {
    if upper < lower {
        let t = upper;
        upper = lower;
        lower = t;
    }

    let mut f_min = f(lower);

    if f_min.abs() < f32::EPSILON {
        return Some(lower);
    }

    let mut f_max = f(upper);

    if f_max.abs() < f32::EPSILON {
        return Some(upper);
    }

    let mut root = 0.5 * (lower + upper);

    if f_min.signum() == f_max.signum() {
        return None;
    }

    for _ in 0..iters {
        let f_root = f(root);

        if upper - lower <= 2.0 * acc && f_root.abs() <= acc {
            return Some(root);
        }

        if (lower - root).abs() < f32::EPSILON || (upper - root).abs() < f32::EPSILON {
            return None;
        }

        if f_root.signum() == f_min.signum() {
            lower = root;
            f_min = f_root;
        } else if f_root.signum() == f_max.signum() {
            upper = root;
            f_max = f_root;
        } else {
            return Some(root);
        }

        root = 0.5 * (lower + upper);
    }

    None
}

pub(crate) fn try_expand_find_root_brent(
    f: impl FnMut(f32) -> f32 + Copy,
    mut lower: f32,
    mut upper: f32,
    acc: f32,
    iters: usize,
    factor: f32,
    expand_iters: usize,
) -> Option<f32> {
    debug_assert!(expand_reduce_for_root(
        f,
        &mut lower,
        &mut upper,
        factor,
        expand_iters,
        expand_iters * 10,
    ));

    try_find_root_brent(f, lower, upper, acc, iters)
}

#[inline]
fn expand_reduce_for_root(
    f: impl FnMut(f32) -> f32 + Copy,
    lower: &mut f32,
    upper: &mut f32,
    factor: f32,
    iters: usize,
    subdivs: usize,
) -> bool {
    expand_for_root(f, lower, upper, factor, iters) || reduce_for_root(f, lower, upper, subdivs)
}

fn expand_for_root(
    mut f: impl FnMut(f32) -> f32,
    lower: &mut f32,
    upper: &mut f32,
    factor: f32,
    iters: usize,
) -> bool {
    let orig_lower = *lower;
    let orig_upper = *upper;

    let mut f_min = f(*lower);
    let mut f_max = f(*upper);

    for _ in 0..iters {
        if f_min.signum() != f_max.signum() {
            return true;
        }

        if f_min.abs() < f_max.abs() {
            *lower += factor * (*lower - *upper);
            f_min = f(*lower);
        } else {
            *upper += factor * (*upper - *lower);
            f_max = f(*upper);
        }
    }

    *lower = orig_lower;
    *upper = orig_upper;

    false
}

fn reduce_for_root(
    mut f: impl FnMut(f32) -> f32,
    lower: &mut f32,
    upper: &mut f32,
    subdivs: usize,
) -> bool {
    let orig_lower = *lower;
    let orig_upper = *upper;

    let f_min = f(*lower);
    let f_max = f(*upper);

    if f_min != f_max {
        return true;
    }

    let subdiv = (*upper - *lower) / subdivs as f32;
    let mut s_min = *lower;
    let sign = f_min.signum();

    for _ in 0..subdivs {
        let s_max = s_min + subdiv;
        let sf_max = f(s_max);

        if sf_max.is_infinite() {
            s_min = s_max;

            continue;
        }

        if sf_max.signum() != sign {
            *lower = s_min;
            *upper = s_max;

            return true;
        }

        s_min = s_max;
    }

    *lower = orig_lower;
    *upper = orig_upper;

    false
}

#[inline]
fn almost_equal_relative(a: f32, b: f32) -> bool {
    almost_equal_norm_relative(a, b, a - b, f32::EPSILON)
}

fn almost_equal_norm_relative(a: f32, b: f32, diff: f32, max_err: f32) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }

    if a.is_nan() || b.is_nan() {
        return false;
    }

    if a.abs() < f32::EPSILON || b.abs() < f32::EPSILON {
        return diff.abs() < max_err;
    }

    if b.abs() < max_err || a.abs() < max_err {
        return true;
    }

    diff.abs() < max_err * a.abs().max(b.abs())
}

#[inline]
fn cdf(mean: f32, std_dev: f32, x: f32) -> f32 {
    0.5 * erfc((mean - x) / (std_dev * SQRT_2))
}

#[inline]
fn pdf(mean: f32, std_dev: f32, x: f32) -> f32 {
    let d = (x - mean) / std_dev;

    (-0.5 * d * d).exp() / ((2.0 * PI).sqrt() * std_dev)
}

pub(crate) struct PoissonBinomial {
    mu: f32,
    sigma: f32,
    v: f32,
}

impl PoissonBinomial {
    pub(crate) fn new(probabilities: impl Iterator<Item = f32>) -> Option<Self> {
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut gamma = 0.0;

        for p in probabilities {
            mu += p;
            sigma += p * (1.0 - p);
            gamma += p * (1.0 - p) * (1.0 - 2.0 * p);
        }

        if mu.abs() < f32::EPSILON {
            return None;
        }

        sigma = sigma.sqrt();
        let v = gamma / (6.0 * sigma * sigma * sigma);

        Some(Self { mu, sigma, v })
    }

    // pub(crate) fn new(probabilities: &[f32]) -> Self {
    //     let mu = probabilities.iter().sum::<f32>();
    //     let mut sigma = 0.0;
    //     let mut gamma = 0.0;

    //     for p in probabilities {
    //         sigma += p * (1.0 - p);
    //         gamma += p * (1.0 - p) * (1.0 - 2.0 * p);
    //     }

    //     sigma = sigma.sqrt();
    //     let v = gamma / (6.0 * sigma * sigma * sigma);

    //     Self { mu, sigma, v }
    // }

    #[inline]
    pub(crate) fn cdf(&self, count: f32) -> f32 {
        if self.sigma.abs() < f32::EPSILON {
            return 1.0;
        }

        let k = (count + 0.5 - self.mu) / self.sigma;
        let result = cdf(0.0, 1.0, k) + self.v * (1.0 - k * k) * pdf(0.0, 1.0, k);

        result.clamp(0.0, 1.0)
    }
}

pub(crate) fn linear_spaced(len: usize, start: f32, end: f32) -> Vec<f32> {
    if len == 0 {
        return vec![0.0];
    } else if len == 1 {
        return vec![end];
    }

    let step = (end - start) / (len - 1) as f32;

    (0..len).map(|i| step.mul_add(i as f32, start)).collect()
}

#[inline]
pub(crate) fn pow_mean(x: f32, y: f32, i: f32) -> f32 {
    ((x.powf(i) + y.powf(i)) / 2.0).powf(i.recip())
}
