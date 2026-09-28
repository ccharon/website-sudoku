//! JavaScript arithmetic where Rust rounds differently.

/// Math.round: halves go up, also below zero.
pub fn round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 { f + 1.0 } else { f }
}

pub fn round_i(x: f64) -> isize {
    round(x) as isize
}

/// A store into a Uint8ClampedArray: halves go to the even neighbour.
pub fn clamp_u8(x: f64) -> u8 {
    if x.is_nan() || x <= 0.0 {
        0
    } else if x >= 255.0 {
        255
    } else {
        x.round_ties_even() as u8
    }
}

/// Math.hypot as V8 computes it, which can differ from libm in the last bit.
pub fn hypot(a: f64, b: f64) -> f64 {
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let values = [a.abs(), b.abs()];
    let max = values[0].max(values[1]);
    if max == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for v in values {
        let n = v / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// Math.min over several values, NaN as soon as one is NaN.
pub fn min(values: &[f64]) -> f64 {
    values.iter().fold(f64::INFINITY, |m, &v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.min(v) })
}

pub fn max(values: &[f64]) -> f64 {
    values.iter().fold(f64::NEG_INFINITY, |m, &v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_like_javascript() {
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(-2.5), -2.0);
        assert_eq!(round(0.49999999999999994), 0.0);
        assert_eq!(clamp_u8(2.5), 2);
        assert_eq!(clamp_u8(3.5), 4);
        assert_eq!(clamp_u8(f64::NAN), 0);
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert!(min(&[1.0, f64::NAN]).is_nan());
    }
}
