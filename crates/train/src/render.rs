//! Synthetic photos of printed Sudoku puzzles with known digits.

use std::fs;
use std::path::Path;

use ab_glyph::{Font, FontVec, PxScale, point};
use sudoku_ocr::Quad;

use crate::rng::Rng;

/// File name prefixes of the font families drawn from, one family each.
pub const FAMILIES: [&str; 15] = [
    "LiberationSans-",
    "LiberationSerif-",
    "LiberationMono-",
    "NimbusSans-",
    "NimbusSansNarrow-",
    "NimbusRoman-",
    "NimbusMonoPS-",
    "C059-",
    "P052-",
    "URWBookman-",
    "URWGothic-",
    "Hack-",
    "NotoSans-",
    "NotoSerif-",
    "NotoSansMono-",
];

pub struct Face {
    pub name: String,
    font: FontVec,
    /// Height of the digit 8 at a scale of one pixel per em.
    digit_height: f32,
}

/// The fonts of every family found under dir, grouped by family.
pub fn load_families(dir: &Path, families: &[&str]) -> Vec<Vec<Face>> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "ttf") {
                files.push(path);
            }
        }
    }
    files.sort();
    families
        .iter()
        .map(|prefix| {
            files
                .iter()
                .filter(|f| f.file_name().is_some_and(|n| n.to_string_lossy().starts_with(prefix)))
                .filter_map(|f| {
                    let font = FontVec::try_from_vec(fs::read(f).ok()?).ok()?;
                    let glyph = font.glyph_id('8').with_scale_and_position(100.0, point(0.0, 0.0));
                    let height = font.outline_glyph(glyph)?.px_bounds().height() / 100.0;
                    let name = f.file_name()?.to_string_lossy().into_owned();
                    Some(Face { name, font, digit_height: height })
                })
                .collect()
        })
        .collect()
}

/// A grey image, 0 black and 1 white.
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Gray {
    fn new(w: usize, h: usize, tone: f32) -> Self {
        Gray { w, h, px: vec![tone; w * h] }
    }

    fn blend(&mut self, x: usize, y: usize, tone: f32, cover: f32) {
        let p = &mut self.px[y * self.w + x];
        *p += (tone - *p) * cover.clamp(0.0, 1.0);
    }

    /// A rectangle with edges covering partial pixels.
    fn fill_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, tone: f32) {
        let span = |a: f64, b: f64, p: usize| ((p as f64 + 1.0).min(b) - (p as f64).max(a)).max(0.0);
        let px0 = x0.floor().max(0.0) as usize;
        let py0 = y0.floor().max(0.0) as usize;
        let px1 = (x1.ceil().max(0.0) as usize).min(self.w);
        let py1 = (y1.ceil().max(0.0) as usize).min(self.h);
        for y in py0..py1 {
            let cy = span(y0, y1, y);
            for x in px0..px1 {
                self.blend(x, y, tone, (cy * span(x0, x1, x)) as f32);
            }
        }
    }

    /// A digit of the given height, centred on (cx, cy).
    fn draw_digit(&mut self, face: &Face, digit: u8, height: f64, cx: f64, cy: f64, tone: f32) {
        let scale = PxScale::from((height / face.digit_height as f64) as f32);
        let id = face.font.glyph_id((b'0' + digit) as char);
        let Some(probe) = face.font.outline_glyph(id.with_scale_and_position(scale, point(0.0, 0.0))) else { return };
        let b = probe.px_bounds();
        let at = point(cx as f32 - (b.min.x + b.max.x) / 2.0, cy as f32 - (b.min.y + b.max.y) / 2.0);
        let Some(glyph) = face.font.outline_glyph(id.with_scale_and_position(scale, at)) else { return };
        let b = glyph.px_bounds();
        glyph.draw(|x, y, cover| {
            let (px, py) = (b.min.x as i64 + x as i64, b.min.y as i64 + y as i64);
            if px >= 0 && py >= 0 && (px as usize) < self.w && (py as usize) < self.h {
                self.blend(px as usize, py as usize, tone, cover);
            }
        });
    }
}

/// A printed puzzle: the sheet and where the grid lies on it.
pub struct Sheet {
    pub image: Gray,
    /// Outer corners of the grid on the sheet, in pixels, on the middle of the frame.
    pub grid: Quad,
    /// Width of the frame around the grid, in cells.
    pub frame: f64,
}

/// Draws the givens of a puzzle into a grid. Empty cells may carry small
/// pencil marks, which the reader has to take for no digit.
pub fn draw_sheet(rng: &mut Rng, face: &Face, marks: &Face, puzzle: &[u8; 81]) -> Sheet {
    let cell = rng.range(26.0, 64.0);
    let pad = cell * rng.range(0.3, 1.6);
    let side = (9.0 * cell + 2.0 * pad).ceil() as usize;
    let paper = rng.range(0.82, 1.0) as f32;
    let ink = rng.range(0.0, 0.3) as f32;
    let mut img = Gray::new(side, side, paper);

    let thin = rng.range(0.6, 2.5);
    let thick = if rng.chance(0.25) {
        cell * rng.range(0.1, 0.45)
    } else if rng.chance(0.15) {
        thin
    } else {
        thin * rng.range(1.5, 4.0)
    };
    let outer = thick * rng.range(1.0, 1.6);
    let dotted = rng.chance(0.12);
    let line_tone = if rng.chance(0.3) { ink + rng.range(0.1, 0.4) as f32 } else { ink };
    let (from, to) = (pad, pad + 9.0 * cell);
    for i in 0..=9 {
        let at = pad + i as f64 * cell;
        let wd = if i == 0 || i == 9 { outer } else if i % 3 == 0 { thick } else { thin };
        if dotted && i % 3 != 0 {
            let dash = cell * 0.08;
            let mut s = from;
            while s < to {
                img.fill_rect(at - wd / 2.0, s, at + wd / 2.0, (s + dash).min(to), line_tone);
                img.fill_rect(s, at - wd / 2.0, (s + dash).min(to), at + wd / 2.0, line_tone);
                s += 2.0 * dash;
            }
        } else {
            img.fill_rect(at - wd / 2.0, from - outer / 2.0, at + wd / 2.0, to + outer / 2.0, line_tone);
            img.fill_rect(from - outer / 2.0, at - wd / 2.0, to + outer / 2.0, at + wd / 2.0, line_tone);
        }
    }

    let height = cell * rng.range(0.42, 0.8);
    let jitter = cell * 0.06;
    let pencil = rng.chance(0.25);
    for (i, &d) in puzzle.iter().enumerate() {
        let (r, c) = ((i / 9) as f64, (i % 9) as f64);
        let (cx, cy) = (pad + (c + 0.5) * cell, pad + (r + 0.5) * cell);
        if d != 0 {
            let jx = rng.range(-jitter, jitter);
            let jy = rng.range(-jitter, jitter);
            img.draw_digit(face, d, height, cx + jx, cy + jy, ink);
        } else if pencil && rng.chance(0.4) {
            for m in 1..=9u8 {
                if rng.chance(0.3) {
                    let (mr, mc) = (((m - 1) / 3) as f64, ((m - 1) % 3) as f64);
                    let (mx, my) = (cx + (mc - 1.0) * cell * 0.3, cy + (mr - 1.0) * cell * 0.3);
                    img.draw_digit(marks, m, cell * rng.range(0.14, 0.24), mx, my, ink + 0.2);
                }
            }
        }
    }
    Sheet { image: img, grid: [[from, from], [to, from], [to, to], [from, to]], frame: outer / cell }
}

/// A homography from the unit square onto a quadrilateral, and back.
struct Homography([f64; 9]);

impl Homography {
    /// After Heckbert, as unitToQuad in camera.js.
    fn unit_to(q: &Quad) -> Self {
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *q;
        let (dx1, dx2, dx3) = (x1 - x2, x3 - x2, x0 - x1 + x2 - x3);
        let (dy1, dy2, dy3) = (y1 - y2, y3 - y2, y0 - y1 + y2 - y3);
        let det = dx1 * dy2 - dy1 * dx2;
        let g = (dx3 * dy2 - dy3 * dx2) / det;
        let h = (dx1 * dy3 - dy1 * dx3) / det;
        Homography([x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h, 1.0])
    }

    fn inverse(&self) -> Self {
        let [a, b, c, d, e, f, g, h, i] = self.0;
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        Homography([
            (e * i - f * h) / det,
            (c * h - b * i) / det,
            (b * f - c * e) / det,
            (f * g - d * i) / det,
            (a * i - c * g) / det,
            (c * d - a * f) / det,
            (d * h - e * g) / det,
            (b * g - a * h) / det,
            (a * e - b * d) / det,
        ])
    }

    fn map(&self, x: f64, y: f64) -> [f64; 2] {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w]
    }
}

/// An RGBA photo and where the grid lies in it.
pub struct Shot {
    pub rgba: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub grid: Quad,
    /// Light ink on a dark ground.
    pub dark: bool,
}

fn gaussian_blur(img: &mut Gray, sigma: f64) {
    let radius = (sigma * 3.0).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius).map(|k| (-(k * k) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / sum).collect();
    let (w, h) = (img.w as isize, img.h as isize);
    for horizontal in [true, false] {
        let src = img.px.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, wk) in kernel.iter().enumerate() {
                    let o = k as isize - radius;
                    let (sx, sy) = if horizontal { ((x + o).clamp(0, w - 1), y) } else { (x, (y + o).clamp(0, h - 1)) };
                    acc += wk * src[(sy * w + sx) as usize];
                }
                img.px[(y * w + x) as usize] = acc;
            }
        }
    }
}

/// Photographs the sheet: perspective, blur, uneven light, noise, low
/// contrast, sometimes a dark table around it or a screen in dark mode.
pub fn photograph(rng: &mut Rng, sheet: &Sheet) -> Shot {
    let src = &sheet.image;
    let flat = rng.chance(0.2);
    let (w, h, to_photo) = if flat {
        (src.w, src.h, Homography([src.w as f64, 0.0, 0.0, 0.0, src.h as f64, 0.0, 0.0, 0.0, 1.0]))
    } else {
        let w = rng.range(800.0, 1400.0) as usize;
        let h = (w as f64 * rng.range(0.8, 1.3)) as usize;
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let r = w.min(h) as f64 * rng.range(0.55, 0.92) / 2.0;
        let angle = rng.range(-0.25, 0.25);
        let warp = rng.range(0.0, 0.03);
        let tilt = rng.range(-0.12, 0.12);
        let (cos, sin) = (angle.cos(), angle.sin());
        let corners = [[-r, -r], [r, -r], [r, r], [-r, r]];
        let quad: Quad = std::array::from_fn(|k| {
            let jx = rng.range(-1.0, 1.0) * r * warp + if k < 2 { -tilt * r } else { tilt * r };
            let jy = rng.range(-1.0, 1.0) * r * warp;
            let (x, y) = (corners[k][0] + jx, corners[k][1] + jy);
            [cx + x * cos - y * sin, cy + x * sin + y * cos]
        });
        (w, h, Homography::unit_to(&quad))
    };

    let background = if rng.chance(0.25) { rng.range(0.02, 0.25) } else { rng.range(0.5, 0.95) } as f32;
    let mut img = Gray::new(w, h, background);
    let back = to_photo.inverse();
    for y in 0..h {
        for x in 0..w {
            let [u, v] = back.map(x as f64 + 0.5, y as f64 + 0.5);
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let (sx, sy) = (u * src.w as f64 - 0.5, v * src.h as f64 - 0.5);
            let (x0, y0) = (sx.floor().max(0.0) as usize, sy.floor().max(0.0) as usize);
            let (x1, y1) = ((x0 + 1).min(src.w - 1), (y0 + 1).min(src.h - 1));
            let (fx, fy) = ((sx - x0 as f64).clamp(0.0, 1.0) as f32, (sy - y0 as f64).clamp(0.0, 1.0) as f32);
            let at = |xx: usize, yy: usize| src.px[yy * src.w + xx];
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            img.px[y * w + x] = top + (bottom - top) * fy;
        }
    }

    let sigma = if rng.chance(0.3) { 0.0 } else { rng.range(0.3, 1.6) };
    if sigma > 0.0 {
        gaussian_blur(&mut img, sigma);
    }
    let shade = rng.range(0.0, 0.4) as f32;
    let light = rng.range(0.0, std::f64::consts::TAU);
    let (lx, ly) = (light.cos() as f32, light.sin() as f32);
    let noise = rng.range(0.0, 12.0) as f32 / 255.0;
    let contrast = rng.range(0.6, 1.0) as f32;
    let dark = rng.chance(0.1);
    let mut rgba = vec![255; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let gain = 1.0 - shade * ((x as f32 / w as f32 - 0.5) * lx + (y as f32 / h as f32 - 0.5) * ly + 0.5);
            let mut v = img.px[y * w + x] * gain + rng.normal() as f32 * noise;
            v = 0.5 + (v - 0.5) * contrast;
            if dark {
                v = 1.0 - v;
            }
            let byte = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            rgba[(y * w + x) * 4..][..3].fill(byte);
        }
    }

    let grid = sheet.grid.map(|p| to_photo.map(p[0] / src.w as f64, p[1] / src.h as f64));
    Shot { rgba, w, h, grid, dark }
}
