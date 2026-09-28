//! A small convolutional network that reads one normalised cell.
//!
//! Three blocks of 3×3 convolution, ReLU and 2×2 max pooling take the 32×32
//! mask down to 32 channels of 4×4. A dense layer maps those, together with
//! the shape measures of the cell, to ten classes: 0 for "no digit" and the
//! digits 1 to 9.
//!
//! The weights are one flat array of f32, little endian, in the order conv1
//! weights, conv1 biases, conv2 weights, conv2 biases, conv3 weights, conv3
//! biases, dense weights, dense biases. A convolution stores its weights as
//! [out][in][3][3], the dense layer as [class][input].

use crate::cells::{NORM, SHAPE};

pub const CLASSES: usize = 10;

/// Input channels, output channels and side length of each convolution.
pub const CONVS: [(usize, usize, usize); 3] = [(1, 8, NORM), (8, 16, NORM / 2), (16, 32, NORM / 4)];

/// Outputs of the last convolution after pooling.
pub const POOLED: usize = 32 * (NORM / 8) * (NORM / 8);

/// Inputs of the dense layer: the pooled convolution, then the shape measures.
pub const DENSE_IN: usize = POOLED + SHAPE;

/// Where each layer's weights and biases start in the flat parameter array.
pub struct Layout {
    pub conv_w: [usize; 3],
    pub conv_b: [usize; 3],
    pub dense_w: usize,
    pub dense_b: usize,
    pub len: usize,
}

pub const LAYOUT: Layout = {
    let mut at = 0;
    let mut conv_w = [0; 3];
    let mut conv_b = [0; 3];
    let mut i = 0;
    while i < 3 {
        let (cin, cout, _) = CONVS[i];
        conv_w[i] = at;
        at += cout * cin * 9;
        conv_b[i] = at;
        at += cout;
        i += 1;
    }
    let dense_w = at;
    at += CLASSES * DENSE_IN;
    let dense_b = at;
    at += CLASSES;
    Layout { conv_w, conv_b, dense_w, dense_b, len: at }
};

/// Everything the forward pass computed, which training needs to go back.
pub struct Trace {
    /// The input of each convolution.
    pub input: [Vec<f32>; 3],
    /// Each convolution after ReLU, before pooling.
    pub relu: [Vec<f32>; 3],
    /// The input of the dense layer.
    pub flat: Vec<f32>,
    pub logits: [f32; CLASSES],
}

pub struct Net {
    pub params: Vec<f32>,
}

impl Net {
    pub fn from_le_bytes(bytes: &[u8]) -> Option<Net> {
        if bytes.len() != LAYOUT.len * 4 {
            return None;
        }
        let params = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Some(Net { params })
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.params.iter().flat_map(|p| p.to_le_bytes()).collect()
    }

    pub fn trace(&self, mask: &[u8], shape: &[f32; SHAPE]) -> Trace {
        let mut x: Vec<f32> = mask.iter().map(|&v| v as f32).collect();
        let mut input: [Vec<f32>; 3] = Default::default();
        let mut relu: [Vec<f32>; 3] = Default::default();
        for (i, &(cin, cout, side)) in CONVS.iter().enumerate() {
            let w = &self.params[LAYOUT.conv_w[i]..LAYOUT.conv_b[i]];
            let b = &self.params[LAYOUT.conv_b[i]..LAYOUT.conv_b[i] + cout];
            let r = conv_relu(&x, cin, cout, side, w, b);
            let pooled = max_pool(&r, cout, side);
            input[i] = x;
            relu[i] = r;
            x = pooled;
        }
        x.extend_from_slice(shape);
        let w = &self.params[LAYOUT.dense_w..LAYOUT.dense_b];
        let b = &self.params[LAYOUT.dense_b..LAYOUT.len];
        let logits = std::array::from_fn(|k| {
            b[k] + w[k * DENSE_IN..(k + 1) * DENSE_IN].iter().zip(&x).map(|(w, x)| w * x).sum::<f32>()
        });
        Trace { input, relu, flat: x, logits }
    }

    /// The probability of each class for one 32×32 mask and its shape measures.
    pub fn classify(&self, mask: &[u8], shape: &[f32; SHAPE]) -> [f32; CLASSES] {
        softmax(&self.trace(mask, shape).logits)
    }
}

pub fn softmax(logits: &[f32; CLASSES]) -> [f32; CLASSES] {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e = logits.map(|l| (l - top).exp());
    let sum: f32 = e.iter().sum();
    e.map(|v| v / sum)
}

/// 3×3 convolution with zero padding, then ReLU. Channels first.
fn conv_relu(x: &[f32], cin: usize, cout: usize, side: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = side * side;
    let mut out = vec![0.0; cout * n];
    for o in 0..cout {
        let plane = &mut out[o * n..(o + 1) * n];
        plane.fill(b[o]);
        for i in 0..cin {
            let src = &x[i * n..(i + 1) * n];
            let k = &w[(o * cin + i) * 9..][..9];
            for ky in 0..3 {
                for kx in 0..3 {
                    let wk = k[ky * 3 + kx];
                    // Rows and columns of the output whose source pixel lies inside.
                    let (y0, y1) = (1usize.saturating_sub(ky), (side + 1 - ky).min(side));
                    let (x0, x1) = (1usize.saturating_sub(kx), (side + 1 - kx).min(side));
                    for y in y0..y1 {
                        let sy = y + ky - 1;
                        let row = &mut plane[y * side..(y + 1) * side];
                        let srow = &src[sy * side..(sy + 1) * side];
                        for xx in x0..x1 {
                            row[xx] += wk * srow[xx + kx - 1];
                        }
                    }
                }
            }
        }
        for v in plane.iter_mut() {
            *v = v.max(0.0);
        }
    }
    out
}

/// 2×2 max pooling, halving each side.
fn max_pool(x: &[f32], channels: usize, side: usize) -> Vec<f32> {
    let half = side / 2;
    let mut out = vec![0.0; channels * half * half];
    for c in 0..channels {
        for y in 0..half {
            for xx in 0..half {
                let at = |dy: usize, dx: usize| x[c * side * side + (2 * y + dy) * side + 2 * xx + dx];
                out[c * half * half + y * half + xx] = at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1));
            }
        }
    }
    out
}
