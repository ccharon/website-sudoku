/// Returns the luma plane of RGBA pixels.
pub fn to_gray(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|[r, g, b, _]| ((*r as u32 * 77 + *g as u32 * 150 + *b as u32 * 29) >> 8) as u8)
        .collect()
}

/// The grey plane with light and dark exchanged, for a screen in dark mode.
pub fn invert_plane(gray: &[u8]) -> Vec<u8> {
    gray.iter().map(|&g| 255 - g).collect()
}

/// Sums of a grey plane, one row and column larger than the image.
fn integral(gray: &[u8], w: usize, h: usize) -> Vec<f64> {
    let stride = w + 1;
    let mut sum = vec![0.0; stride * (h + 1)];
    for y in 0..h {
        let mut row = 0.0;
        for x in 0..w {
            row += gray[y * w + x] as f64;
            sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
        }
    }
    sum
}

/// Marks every pixel darker than the mean of its neighbourhood. A global
/// threshold fails on photos because of shadows and uneven light. The pixel
/// must also lie step grey levels below the mean, so noise on a black table
/// does not count as ink.
pub fn threshold(gray: &[u8], w: usize, h: usize, radius: usize, bias: f64, step: f64) -> Vec<u8> {
    let sum = integral(gray, w, h);
    let stride = w + 1;
    let mut mask = vec![0; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(w - 1);
            let count = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            let area = sum[(y1 + 1) * stride + x1 + 1] - sum[y0 * stride + x1 + 1]
                - sum[(y1 + 1) * stride + x0]
                + sum[y0 * stride + x0];
            let g = gray[y * w + x] as f64;
            mask[y * w + x] = (g * count < area * bias && (g + step) * count < area) as u8;
        }
    }
    mask
}

/// A connected patch of ink with its bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Patch {
    pub id: i32,
    pub count: usize,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Patch {
    pub fn width(&self) -> usize {
        self.x1 - self.x0 + 1
    }

    pub fn height(&self) -> usize {
        self.y1 - self.y0 + 1
    }
}

/// Copies a rectangle out of a mask into a plane of its own.
pub fn copy_rect(mask: &[u8], stride: usize, x0: usize, y0: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0; w * h];
    for y in 0..h {
        out[y * w..(y + 1) * w].copy_from_slice(&mask[(y0 + y) * stride + x0..][..w]);
    }
    out
}

/// Labels the connected patches of ink, eight-connected. Returns the label
/// plane, -1 for no ink, and one patch per label.
pub fn components(mask: &[u8], w: usize, h: usize) -> (Vec<i32>, Vec<Patch>) {
    let mut labels = vec![-1; w * h];
    let mut stack = Vec::new();
    let mut patches = Vec::new();
    for start in 0..w * h {
        if mask[start] == 0 || labels[start] >= 0 {
            continue;
        }
        let id = patches.len() as i32;
        let (sx, sy) = (start % w, start / w);
        let mut patch = Patch { id, count: 0, x0: sx, y0: sy, x1: sx, y1: sy };
        stack.push(start);
        labels[start] = id;
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            patch.count += 1;
            patch.x0 = patch.x0.min(x);
            patch.x1 = patch.x1.max(x);
            patch.y0 = patch.y0.min(y);
            patch.y1 = patch.y1.max(y);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let q = ny * w + nx;
                    if mask[q] == 1 && labels[q] < 0 {
                        labels[q] = id;
                        stack.push(q);
                    }
                }
            }
        }
        patches.push(patch);
    }
    (labels, patches)
}

/// Marks the pixels that sit in a run of at least the given length, along the
/// rows and down the columns. A grid line reaches that far, a digit stroke and
/// a speck of dirt do not.
pub fn long_runs(mask: &[u8], w: usize, h: usize, need_x: usize, need_y: usize) -> Vec<u8> {
    let mut out = vec![0; w * h];
    for y in 0..h {
        let mut run = 0;
        for x in 0..=w {
            if x < w && mask[y * w + x] == 1 {
                run += 1;
                continue;
            }
            if run >= need_x {
                out[y * w + x - run..y * w + x].fill(1);
            }
            run = 0;
        }
    }
    for x in 0..w {
        let mut run = 0;
        for y in 0..=h {
            if y < h && mask[y * w + x] == 1 {
                run += 1;
                continue;
            }
            if run >= need_y {
                for k in y - run..y {
                    out[k * w + x] = 1;
                }
            }
            run = 0;
        }
    }
    out
}

/// Copies one labelled patch out of a label plane as a mask of its own.
pub fn patch_mask(labels: &[i32], w: usize, patch: &Patch) -> Vec<u8> {
    let (bw, bh) = (patch.width(), patch.height());
    let mut out = vec![0; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            out[y * bw + x] = (labels[(patch.y0 + y) * w + patch.x0 + x] == patch.id) as u8;
        }
    }
    out
}
