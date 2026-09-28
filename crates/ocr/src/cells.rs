use crate::ink::{Patch, components, copy_rect, long_runs};
use crate::js;

const CELL_INSET: f64 = 0.14; // share of a cell dropped on each side
const CELL_OFF: f64 = 0.36; // how far off centre a digit may sit inside a cell
const CELL_SPAN: f64 = 0.92; // a run reaching this far across a cell is a grid line
const CELL_FLAT: f64 = 0.22; // shorter than this share of a cell it is a piece of a line
const SPECK: usize = 5; // pixels, less than that is threshold noise
const INK_ANY: f64 = 0.006; // share of the cell below which it counts as empty
const INK_FAINT: f64 = 0.02; // and below which the digit is not to be relied on
const FAINT_PIXELS: usize = 10; // as is a digit drawn from fewer pixels than this
const FAINT_TALL: f64 = 0.3; // or one shorter than this share of the cell
pub const NORM: usize = 32; // side of a normalised digit bitmap
pub const SHAPE: usize = 5; // measures of where and how large the ink sat in its cell
const NORM_FIT: f64 = 24.0; // the digit is scaled to fit this box
const NORM_COVER: f64 = 0.4; // ink a shrunken pixel needs before it is set

/// The digit of one cell, scaled and centred.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub mask: [u8; NORM * NORM],
    /// Too little ink to call a digit. Kept so the caller can still ask.
    pub faint: bool,
    /// What scaling and centring the mask lost: width and height of the ink
    /// over those of the cell, how far its middle sits off the middle of the
    /// cell in the same units, and the share of the cell covered with ink.
    /// A rest of a grid line looks like a 1 once centred, but not here.
    pub shape: [f32; SHAPE],
}

/// Clears the runs that cross the whole cell. Only a grid line reaches that
/// far, and a thick one touches the digit, which would then be dropped along
/// with it.
fn strip_spans(mask: &mut [u8], w: usize, h: usize) {
    let need_x = js::round_i(w as f64 * CELL_SPAN) as usize;
    let need_y = js::round_i(h as f64 * CELL_SPAN) as usize;
    let lines = long_runs(mask, w, h, need_x, need_y);
    for (m, l) in mask.iter_mut().zip(lines) {
        if l == 1 {
            *m = 0;
        }
    }
}

/// Scales the digit to a fixed box and centres it by its centre of mass, so
/// font size and position inside the cell stop mattering.
fn normalise(bitmap: &[u8], w: usize, patch: &Patch) -> Option<[u8; NORM * NORM]> {
    let (bw, bh) = (patch.width(), patch.height());
    let scale = NORM_FIT / bw.max(bh) as f64;
    let fit = |len: usize| js::round_i(len as f64 * scale).clamp(1, NORM as isize) as usize;
    let (tw, th) = (fit(bw), fit(bh));

    let mut small = vec![0u8; tw * th];
    let (mut cx, mut cy, mut count) = (0, 0, 0);
    for y in 0..th {
        let sy0 = y * bh / th;
        let sy1 = ((y + 1) * bh / th).max(sy0 + 1);
        for x in 0..tw {
            let sx0 = x * bw / tw;
            let sx1 = ((x + 1) * bw / tw).max(sx0 + 1);
            let mut on = 0;
            for yy in sy0..sy1 {
                for xx in sx0..sx1 {
                    on += bitmap[(patch.y0 + yy) * w + patch.x0 + xx] as usize;
                }
            }
            let total = (sy1 - sy0) * (sx1 - sx0);
            if on as f64 / total as f64 >= NORM_COVER {
                small[y * tw + x] = 1;
                cx += x;
                cy += y;
                count += 1;
            }
        }
    }
    if count == 0 {
        return None;
    }

    let half = (NORM / 2) as f64;
    let offset = |c: usize, len: usize| {
        js::round_i(half - c as f64 / count as f64).max(0).min((NORM - len) as isize) as usize
    };
    let (ox, oy) = (offset(cx, tw), offset(cy, th));
    let mut out = [0; NORM * NORM];
    for y in 0..th {
        for x in 0..tw {
            out[(y + oy) * NORM + x + ox] = small[y * tw + x];
        }
    }
    Some(out)
}

/// Returns the normalised digit of one cell, or None if it is empty.
fn isolate_digit(ink: &[u8], size: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Option<Cell> {
    let (w, h) = (x1 - x0, y1 - y0);
    let area = (w * h) as f64;
    let (wf, hf) = (w as f64, h as f64);
    let mut sub = copy_rect(ink, size, x0, y0, w, h);
    strip_spans(&mut sub, w, h);
    let marked: usize = sub.iter().map(|&v| v as usize).sum();
    if (marked as f64) < area * INK_ANY {
        return None;
    }

    let (labels, patches) = components(&sub, w, h);
    let mut best: Option<&Patch> = None;
    for p in &patches {
        let (bw, bh) = (p.width() as f64, p.height() as f64);
        let flat = bh < hf * CELL_FLAT; // flat leftovers are grid lines
        let across = bw > wf * CELL_SPAN || bh > hf * CELL_SPAN;
        // A digit sits in the middle, a piece of a dotted rule sits at the edge.
        let off = ((p.x0 + p.x1) as f64 / 2.0 - wf / 2.0).abs() > wf * CELL_OFF
            || ((p.y0 + p.y1) as f64 / 2.0 - hf / 2.0).abs() > hf * CELL_OFF;
        if p.count < SPECK || flat || across || off {
            continue;
        }
        if best.is_none_or(|b| p.count > b.count) {
            best = Some(p);
        }
    }
    let best = best?;

    let faint = (marked as f64) < area * INK_FAINT
        || best.count < FAINT_PIXELS
        || (best.height() as f64) < hf * FAINT_TALL;

    let digit: Vec<u8> = labels.iter().map(|&l| (l == best.id) as u8).collect();
    let mask = normalise(&digit, w, best)?;
    let shape = [
        best.width() as f32 / w as f32,
        best.height() as f32 / h as f32,
        ((best.x0 + best.x1 + 1) as f32 / 2.0 - w as f32 / 2.0) / w as f32,
        ((best.y0 + best.y1 + 1) as f32 / 2.0 - h as f32 / 2.0) / h as f32,
        marked as f32 / area as f32,
    ];
    Some(Cell { mask, faint, shape })
}

/// Cuts the straightened ink mask into 81 cells, row by row. The outer share
/// of every cell is dropped so the grid lines fall away.
pub fn cut_cells(ink: &[u8], size: usize) -> Vec<Option<Cell>> {
    let cell = size as f64 / 9.0;
    let inset = js::round_i(cell * CELL_INSET);
    let at = |k: usize| js::round_i(k as f64 * cell);
    let mut out = Vec::with_capacity(81);
    for r in 0..9 {
        for c in 0..9 {
            out.push(isolate_digit(
                ink,
                size,
                (at(c) + inset) as usize,
                (at(r) + inset) as usize,
                (at(c + 1) - inset) as usize,
                (at(r + 1) - inset) as usize,
            ));
        }
    }
    out
}
