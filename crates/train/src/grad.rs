//! Backpropagation through sudoku_ocr::net and the Adam optimiser.

use sudoku_ocr::net::{CLASSES, CONVS, DENSE_IN, LAYOUT, Net, POOLED, Trace, softmax};

use crate::rng::Rng;

/// Weights drawn after He, biases zero.
pub fn init(rng: &mut Rng) -> Net {
    let mut params = vec![0.0; LAYOUT.len];
    for (i, &(cin, cout, _)) in CONVS.iter().enumerate() {
        let std = (2.0 / (cin * 9) as f64).sqrt();
        for p in &mut params[LAYOUT.conv_w[i]..LAYOUT.conv_w[i] + cout * cin * 9] {
            *p = (rng.normal() * std) as f32;
        }
    }
    let std = (1.0 / DENSE_IN as f64).sqrt();
    for p in &mut params[LAYOUT.dense_w..LAYOUT.dense_b] {
        *p = (rng.normal() * std) as f32;
    }
    Net { params }
}

/// Adds the gradient of the cross entropy for one sample to grad and
/// returns the loss.
pub fn backward(net: &Net, t: &Trace, label: usize, grad: &mut [f32]) -> f32 {
    let p = softmax(&t.logits);
    let loss = -p[label].max(1e-12).ln();
    let mut d = p;
    d[label] -= 1.0;

    let w = &net.params[LAYOUT.dense_w..LAYOUT.dense_b];
    // The shape measures are inputs, so only the pooled part passes the gradient on.
    let mut dx = vec![0.0; POOLED];
    for k in 0..CLASSES {
        grad[LAYOUT.dense_b + k] += d[k];
        let gw = &mut grad[LAYOUT.dense_w + k * DENSE_IN..][..DENSE_IN];
        let wk = &w[k * DENSE_IN..][..DENSE_IN];
        for (g, x) in gw.iter_mut().zip(&t.flat) {
            *g += d[k] * x;
        }
        for j in 0..POOLED {
            dx[j] += wk[j] * d[k];
        }
    }

    for l in (0..3).rev() {
        let (cin, cout, side) = CONVS[l];
        let n = side * side;
        let half = side / 2;
        let relu = &t.relu[l];

        // Pooling sends the gradient to the largest of the four, ReLU drops it where the output was cut to zero.
        let mut dz = vec![0.0; cout * n];
        for c in 0..cout {
            for y in 0..half {
                for x in 0..half {
                    let g = dx[c * half * half + y * half + x];
                    let mut best = c * n + 2 * y * side + 2 * x;
                    for (dy, dxx) in [(0, 1), (1, 0), (1, 1)] {
                        let at = c * n + (2 * y + dy) * side + 2 * x + dxx;
                        if relu[at] > relu[best] {
                            best = at;
                        }
                    }
                    if relu[best] > 0.0 {
                        dz[best] = g;
                    }
                }
            }
        }

        let input = &t.input[l];
        let wl = &net.params[LAYOUT.conv_w[l]..LAYOUT.conv_b[l]];
        let mut din = if l > 0 { vec![0.0; cin * n] } else { Vec::new() };
        for o in 0..cout {
            let dplane = &dz[o * n..(o + 1) * n];
            grad[LAYOUT.conv_b[l] + o] += dplane.iter().sum::<f32>();
            for i in 0..cin {
                let src = &input[i * n..(i + 1) * n];
                let base = (o * cin + i) * 9;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (y0, y1) = (1usize.saturating_sub(ky), (side + 1 - ky).min(side));
                        let (x0, x1) = (1usize.saturating_sub(kx), (side + 1 - kx).min(side));
                        let wk = wl[base + ky * 3 + kx];
                        let mut gw = 0.0;
                        for y in y0..y1 {
                            let sy = y + ky - 1;
                            for x in x0..x1 {
                                let g = dplane[y * side + x];
                                gw += g * src[sy * side + x + kx - 1];
                                if l > 0 {
                                    din[i * n + sy * side + x + kx - 1] += wk * g;
                                }
                            }
                        }
                        grad[LAYOUT.conv_w[l] + base + ky * 3 + kx] += gw;
                    }
                }
            }
        }
        dx = din;
    }
    loss
}

pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    step: i32,
}

impl Adam {
    pub fn new(len: usize) -> Self {
        Adam { m: vec![0.0; len], v: vec![0.0; len], step: 0 }
    }

    pub fn update(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        self.step += 1;
        let c1 = 1.0 - B1.powi(self.step);
        let c2 = 1.0 - B2.powi(self.step);
        for i in 0..params.len() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grad[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grad[i] * grad[i];
            params[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic gradient agrees with a central difference, for a few
    /// parameters of every layer.
    #[test]
    fn gradient_matches_difference() {
        let mut rng = Rng::new(3);
        let mut net = init(&mut rng);
        for p in &mut net.params[LAYOUT.conv_b[0]..LAYOUT.conv_b[0] + 8] {
            *p = 0.05;
        }
        let mask: Vec<u8> = (0..1024).map(|_| rng.chance(0.3) as u8).collect();
        let shape = [0.5, 0.7, 0.1, -0.05, 0.2];
        let label = 7;
        let mut grad = vec![0.0; LAYOUT.len];
        backward(&net, &net.trace(&mask, &shape), label, &mut grad);

        let loss = |net: &Net| -(softmax(&net.trace(&mask, &shape).logits)[label] as f64).ln();
        let base = loss(&net);
        let starts = [LAYOUT.conv_w[0], LAYOUT.conv_b[0], LAYOUT.conv_w[1], LAYOUT.conv_w[2], LAYOUT.conv_b[2], LAYOUT.dense_w, LAYOUT.dense_w + POOLED, LAYOUT.dense_b];
        for start in starts {
            let mut checked = 0;
            for i in (start..LAYOUT.len).take(20) {
                let keep = net.params[i];
                let h = 1e-3;
                net.params[i] = keep + h;
                let up = (loss(&net) - base) / h as f64;
                net.params[i] = keep - h;
                let down = (base - loss(&net)) / h as f64;
                net.params[i] = keep;
                // Where the two sides disagree a ReLU or pooling kink lies within h.
                if (up - down).abs() > 1e-3 + 0.05 * up.abs() {
                    continue;
                }
                let numeric = (up + down) / 2.0;
                let analytic = grad[i] as f64;
                assert!((numeric - analytic).abs() < 1e-3 + 0.05 * numeric.abs(), "param {i}: numeric {numeric}, analytic {analytic}");
                checked += 1;
            }
            assert!(checked >= 5, "only {checked} smooth parameters from {start}");
        }
    }
}

