use std::f32::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

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

#[allow(clippy::excessive_precision)]
pub(crate) fn erfcinv(p: f32) -> f32 {
    if p >= 2.0 {
        return -100.0;
    } else if p <= 0.0 {
        return 100.0;
    }

    let pp = if p < 1.0 { p } else { 2.0 - p };
    let t = (-2.0 * (pp / 2.0).ln()).sqrt();
    let mut x =
        -FRAC_1_SQRT_2 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);

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

#[allow(clippy::excessive_precision)]
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
