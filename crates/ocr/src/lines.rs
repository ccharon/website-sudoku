use crate::geometry::{Quad, Rect, median, side_lengths};
use crate::js;

const SIZE: f64 = 9.0;

const LINE_GAP: f64 = 0.03; // share of the image a line may be broken for, dotted rules included
const LINE_COVER: f64 = 0.6; // share of the way across a line has to reach to count
const LINE_THICK: f64 = 0.04; // a band wider than this share of the image is not one line
const MIN_BANDS: usize = 8; // lines a direction must show before it is rated at all
const BOX: usize = 3; // cells along one side of a box
const CELLS_OFF: f64 = 2.0; // how far the counted cells may differ from nine
const DRIFT_COST: f64 = 4.0; // weight of uneven spacing against the rating
const OFF_SIZE: f64 = 0.6; // what a grid that is not nine cells wide still scores
const BOXES_ONLY: f64 = 0.75; // and what one scores that shows only the lines around its boxes
pub const BOUND_KEEP: f64 = 0.5; // share of the image a cut has to leave standing
const BOUND_EDGE: f64 = 0.02; // a bound this close to the edge has cut nothing off
const MIN_CUT_BANDS: usize = 4; // lines needed before the inner extent means anything
const PAPER_RANK: f64 = 0.95; // brightness rank taken as the colour of the paper
const DARK_LEVEL: f64 = 0.5; // darker than this share of the paper is not paper
const FRAME_LINE: f64 = 0.12; // share of a cell up to which a dark edge is the frame and stays
const EDGE_SKIP: f64 = 0.1; // share of each edge left out at the corners
const EDGE_NEAR: f64 = 0.1; // share of a cell a point may lie off the fitted edge

/// Index steps of one direction: from one band to the next, and from one
/// pixel of a line to the next along it. With along set the bands are columns.
fn line_steps(size: usize, along: bool) -> (usize, usize) {
    if along { (1, size) } else { (size, 1) }
}

/// Centres of the runs of pixels that carry a line across the whole image.
fn line_bands(ink: &[u8], size: usize, along: bool) -> Vec<f64> {
    let (band, walk) = line_steps(size, along);
    let n = size as f64;
    let mut bands = Vec::new();
    let mut start = None;
    // The step past the end closes a band that runs into the edge.
    for i in 0..=size {
        let is_line = i < size && {
            let count: u32 = (0..size).map(|k| ink[i * band + k * walk] as u32).sum();
            count as f64 >= n * LINE_COVER
        };
        match (is_line, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if (i - s) as f64 <= n * LINE_THICK {
                    bands.push((s + i - 1) as f64 / 2.0);
                }
                start = None;
            }
            _ => {}
        }
    }
    bands
}

/// Length and position of the longest unbroken stretch of the line through
/// this band. Plain first and last ink would catch every crossing line instead.
fn band_extent(ink: &[u8], size: usize, centre: f64, along: bool) -> Option<(usize, usize)> {
    let (band, walk) = line_steps(size, along);
    let c = js::round_i(centre);
    let from = (c - 1).max(0) as usize;
    let to = (c + 1).min(size as isize - 1) as usize;
    let mut best: Option<(usize, usize)> = None;
    let mut first: Option<usize> = None;
    let mut last = 0;
    let mut gap = 0;

    let mut close = |first: &mut Option<usize>, last: usize| {
        if let Some(f) = *first {
            let longer = match best {
                None => true,
                Some((bf, bl)) => last - f > bl - bf,
            };
            if longer {
                best = Some((f, last));
            }
        }
        *first = None;
    };

    for k in 0..size {
        let hit = (from..=to).any(|i| ink[i * band + k * walk] != 0);
        if hit {
            first.get_or_insert(k);
            last = k;
            gap = 0;
        } else if first.is_some() {
            gap += 1;
            if gap as f64 > size as f64 * LINE_GAP {
                close(&mut first, last);
            }
        }
    }
    close(&mut first, last);
    best
}

/// Where the inner lines of one direction begin and end, across it.
fn inner_extent(ink: &[u8], size: usize, bands: &[f64], along: bool) -> Option<(f64, f64)> {
    let inner = if bands.len() >= 2 { &bands[1..bands.len() - 1] } else { &[] };
    let mut firsts = Vec::new();
    let mut lasts = Vec::new();
    for &centre in inner {
        if let Some((f, l)) = band_extent(ink, size, centre, along) {
            firsts.push(f as f64);
            lasts.push(l as f64);
        }
    }
    if firsts.is_empty() {
        return None;
    }
    Some((median(&firsts), median(&lasts) + 1.0))
}

/// Rates how much the area inside the quadrilateral looks like a Sudoku: ten
/// evenly spaced lines in both directions, spanning the whole of it. Without
/// this the largest patch wins, which on a tablet is the bezel of the screen.
pub fn band_score(ink: &[u8], size: usize) -> f64 {
    let mut score = 1.0;
    for along in [true, false] {
        let bands = line_bands(ink, size, along);
        if bands.len() < BOX + 1 {
            return 0.0;
        }
        let steps: Vec<f64> = bands.windows(2).map(|p| p[1] - p[0]).collect();
        let step = median(&steps);
        if step.is_nan() || step <= 0.0 {
            return 0.0;
        }
        let reach = bands[bands.len() - 1] - bands[0];
        let cells = js::round(reach / step);
        // Faint cell lines drop out and leave only the lines around the boxes.
        let boxes_only = bands.len() == BOX + 1 && cells == BOX as f64;
        if !boxes_only && bands.len() < MIN_BANDS {
            return 0.0;
        }
        if !boxes_only && (cells - SIZE).abs() > CELLS_OFF {
            return 0.0;
        }
        let mut drift = 0.0;
        for d in &steps {
            drift += (d - step).abs();
        }
        drift /= steps.len() as f64 * step;
        let shape = if cells == SIZE { 1.0 } else if boxes_only { BOXES_ONLY } else { OFF_SIZE };
        score *= (1.0 - DRIFT_COST * drift).max(0.0) * (reach / size as f64) * shape;
    }
    score
}

/// Narrows the straightened image down to the 9x9 grid. Printed puzzles often
/// carry a title box that shares its bottom edge with the grid, so the outline
/// encloses ten rows. The inner lines betray the real extent, because they
/// stop at the grid. Whether the cut is an improvement is decided by the caller.
pub fn grid_bounds(ink: &[u8], size: usize) -> Option<Rect> {
    let n = size as f64;
    let mut r = Rect { x0: 0.0, y0: 0.0, x1: n, y1: n };
    for along in [true, false] {
        let bands = line_bands(ink, size, along);
        if bands.len() < MIN_CUT_BANDS {
            return None;
        }
        let (from, to) = inner_extent(ink, size, &bands, along)?;
        if along {
            (r.y0, r.y1) = (from, to);
        } else {
            (r.x0, r.x1) = (from, to);
        }
    }
    if r.x1 - r.x0 < n * BOUND_KEEP || r.y1 - r.y0 < n * BOUND_KEEP {
        return None;
    }
    let trimmed = r.x0 > n * BOUND_EDGE
        || r.y0 > n * BOUND_EDGE
        || r.x1 < n * (1.0 - BOUND_EDGE)
        || r.y1 < n * (1.0 - BOUND_EDGE);
    trimmed.then_some(r)
}

/// The brightness most of the straightened image stays below, taken as the paper.
fn paper_level(flat: &[u8]) -> f64 {
    let mut counts = [0usize; 256];
    for &v in flat {
        counts[v as usize] += 1;
    }
    let need = flat.len() as f64 * (1.0 - PAPER_RANK);
    let mut level = 255;
    let mut seen = 0;
    while level > 0 {
        seen += counts[level];
        if seen as f64 >= need {
            break;
        }
        level -= 1;
    }
    level as f64
}

/// How deep the dark surround reaches in from one edge, as a line
/// depth = a + b * t across the edge. Grid lines that run into the surround
/// reach deeper, so the line is fitted to the shallow half and whatever lies
/// near it.
fn surround_edge(flat: &[u8], size: usize, dark: f64, along: bool, from: usize, dir: isize) -> Option<(f64, f64)> {
    let at = |t: usize, i: usize| -> f64 {
        let d = (from as isize + dir * i as isize) as usize;
        (if along { flat[t * size + d] } else { flat[d * size + t] }) as f64
    };
    let n = size as f64;
    let skip = js::round_i(n * EDGE_SKIP) as usize;
    let mut points: Vec<[f64; 2]> = Vec::new();
    for t in skip..size.saturating_sub(skip) {
        let mut i = 0;
        // The outermost pixels blend the surround with whatever lies past it.
        while (i as f64) < n * BOUND_EDGE && at(t, i) >= dark {
            i += 1;
        }
        if i as f64 >= n * BOUND_EDGE {
            i = 0;
        } else {
            while i < size && at(t, i) < dark {
                i += 1;
            }
        }
        points.push([t as f64, i as f64]);
    }
    let depths: Vec<f64> = points.iter().map(|p| p[1]).collect();
    if median(&depths) < n * FRAME_LINE / SIZE {
        return None;
    }

    let half = points.len() >> 1;
    let (left, right) = points.split_at(half);
    let column = |ps: &[[f64; 2]], k: usize| median(&ps.iter().map(|p| p[k]).collect::<Vec<_>>());
    let (t0, t1) = (column(left, 0), column(right, 0));
    let (d0, d1) = (column(left, 1), column(right, 1));
    let slope = (d1 - d0) / (t1 - t0);
    let near = n * EDGE_NEAR / SIZE;
    let kept: Vec<&[f64; 2]> = points.iter().filter(|p| (p[1] - d0 - slope * (p[0] - t0)).abs() < near).collect();
    if (kept.len() as f64) < points.len() as f64 / 2.0 {
        return None;
    }

    let (mut st, mut sd, mut stt, mut std) = (0.0, 0.0, 0.0, 0.0);
    for p in &kept {
        st += p[0];
        sd += p[1];
        stt += p[0] * p[0];
        std += p[0] * p[1];
    }
    let k = kept.len() as f64;
    let b = (k * std - st * sd) / (k * stt - st * st);
    Some(((sd - b * st) / k, b))
}

/// Cuts off the dark surround that a light sheet on a black table leaves
/// inside the outline, and a frame thick enough to shift the cells. Returns
/// the four corners of what is left, in pixels of the straightened image.
pub fn paper_quad(flat: &[u8], size: usize) -> Option<Quad> {
    let dark = paper_level(flat) * DARK_LEVEL;
    let top = surround_edge(flat, size, dark, false, 0, 1);
    let bottom = surround_edge(flat, size, dark, false, size - 1, -1);
    let left = surround_edge(flat, size, dark, true, 0, 1);
    let right = surround_edge(flat, size, dark, true, size - 1, -1);
    if top.is_none() && bottom.is_none() && left.is_none() && right.is_none() {
        return None;
    }

    let n = size as f64;
    // Each edge as position = p + q * across, with the far edges counted from size.
    let edge = |line: Option<(f64, f64)>, far: bool| match (line, far) {
        (None, true) => (n, 0.0),
        (None, false) => (0.0, 0.0),
        (Some((p, q)), true) => (n - p, -q),
        (Some(l), false) => l,
    };
    let corner = |rows: (f64, f64), cols: (f64, f64)| {
        let x = (cols.0 + cols.1 * rows.0) / (1.0 - cols.1 * rows.1);
        [x, rows.0 + rows.1 * x]
    };
    let (t, b) = (edge(top, false), edge(bottom, true));
    let (l, r) = (edge(left, false), edge(right, true));
    let quad = [corner(t, l), corner(t, r), corner(b, r), corner(b, l)];
    if js::min(&side_lengths(&quad)) < n * BOUND_KEEP { None } else { Some(quad) }
}
