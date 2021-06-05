#[inline]
pub(crate) fn find_root_brent(
    f: impl FnMut(f32) -> f32,
    lower: f32,
    upper: f32,
    acc: f32,
    iters: usize,
) -> f32 {
    match try_find_root_brent(f, lower, upper, acc, iters) {
        Some(root) => root,
        None => (upper - lower) / 2.0,
    }
}

#[allow(clippy::many_single_char_names)]
fn try_find_root_brent(
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

            return Some(root);
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

#[inline]
pub(crate) fn find_root_bisection(
    f: impl FnMut(f32) -> f32,
    lower: f32,
    upper: f32,
    acc: f32,
    iters: usize,
) -> f32 {
    match try_find_root_bisection(f, lower, upper, acc, iters) {
        Some(root) => root,
        None => (upper - lower) / 2.0,
    }
}

fn try_find_root_bisection(
    mut f: impl FnMut(f32) -> f32,
    mut lower: f32,
    mut upper: f32,
    acc: f32,
    iters: usize,
) -> Option<f32> {
    if upper < lower {
        std::mem::swap(&mut upper, &mut lower);
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
        eprintln!("bad bracketing?");

        return None;
    }

    for _ in 0..iters {
        let f_root = f(root);

        if upper - lower <= 2.0 * acc && f_root.abs() <= acc {
            return Some(root);
        }

        if (lower - root).abs() < f32::EPSILON || (upper - root).abs() < f32::EPSILON {
            eprintln!("accuracy not sufficient, but cannot be improved further");

            return Some(root);
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

    eprintln!("no root after max iterations");

    None
}

#[inline]
pub(crate) fn expand_find_root_brent(
    f: impl FnMut(f32) -> f32 + Copy,
    lower: f32,
    upper: f32,
    acc: f32,
    iters: usize,
    factor: f32,
    expand_iters: usize,
) -> f32 {
    match try_expand_find_root_brent(f, lower, upper, acc, iters, factor, expand_iters) {
        Some(root) => root,
        None => (upper - lower) / 2.0,
    }
}

#[inline]
fn try_expand_find_root_brent(
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

    if (f_min - f_max).abs() >= f32::EPSILON {
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

        if (sf_max.signum() - sign).abs() >= f32::EPSILON {
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
        return (a - b).abs() < f32::EPSILON;
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
