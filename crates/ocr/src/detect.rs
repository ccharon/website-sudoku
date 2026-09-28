use crate::geometry::{
    Point, Quad, convex_hull, cross_lines, distance_to_side, fit_line, max_area_quad, order_quad, plausible_quad,
    rectify, side_of, simplify_hull,
};
use crate::ink::{Patch, components, long_runs, patch_mask, threshold};
use crate::js;
use crate::lines::band_score;
use crate::reading::INK_BIAS;

const CANDIDATES: usize = 6; // ink patches examined before settling on one
const SCORE_CELL: usize = 32; // cell size of the cheap pass that rates a candidate
const WEAK_GRID: f64 = 0.02; // below this the patch is tried again without its attachments
const RUN_SHARE: f64 = 0.3; // share of the patch a run must span to pass as a line
const GOOD_ENOUGH: f64 = 0.8; // a candidate this convincing ends the search
const FIT_PASSES: usize = 2; // how often the sides are fitted and the corners taken anew
const SIDE_BAND: f64 = 0.06; // share of a side length a point may lie off it
const SIDE_POINTS: usize = 12; // points a side needs before a line is fitted to it
const MIN_OUTLINE: usize = 8; // points an outline needs to shape a quadrilateral
const HULL_KEEP: usize = 48; // hull points left for the search over quadrilaterals
const PATCH_AREA: f64 = 0.02; // share of the image a patch must cover to be a candidate
const PATCH_RATIO: f64 = 2.0; // width over height, and the same the other way round
const FILL_MIN: f64 = 0.02; // below this the patch is too sparse to be an outline
const FILL_MAX: f64 = 0.6; // above it the patch is a solid area, not an outline

/// Fits a line to each of the four sides and intersects them. A thin or
/// blurred outline loses its corners, but the long sides survive, so the
/// corner follows from them instead of from a pixel that may not be there.
fn refine_quad(quad: Quad, points: &[Point]) -> Quad {
    let mut current = quad;
    for _ in 0..FIT_PASSES {
        let near = side_of(&current) * SIDE_BAND;
        let mut sides: [Vec<Point>; 4] = Default::default();
        for &p in points {
            let mut at = None;
            let mut best = near;
            for i in 0..4 {
                let d = distance_to_side(p, current[i], current[(i + 1) % 4]);
                if d < best {
                    best = d;
                    at = Some(i);
                }
            }
            if let Some(i) = at {
                sides[i].push(p);
            }
        }
        let mut lines = [[0.0; 3]; 4];
        for (line, side) in lines.iter_mut().zip(&sides) {
            if side.len() < SIDE_POINTS {
                return current;
            }
            *line = fit_line(side);
        }
        let mut next = current;
        for i in 0..4 {
            match cross_lines(lines[(i + 3) % 4], lines[i]) {
                Some(corner) => next[i] = corner,
                None => return current,
            }
        }
        current = next;
    }
    current
}

/// The outermost pixel of each row and column, the only ones that lie on a side.
fn outline_points<T: Copy + PartialEq>(labels: &[T], w: usize, patch: &Patch, id: T) -> Vec<Point> {
    let mut points = Vec::new();
    for y in patch.y0..=patch.y1 {
        let mut hits = (patch.x0..=patch.x1).filter(|&x| labels[y * w + x] == id);
        if let Some(from) = hits.next() {
            let to = hits.last().unwrap_or(from);
            points.push([from as f64, y as f64]);
            if to != from {
                points.push([to as f64, y as f64]);
            }
        }
    }
    for x in patch.x0..=patch.x1 {
        let mut hits = (patch.y0..=patch.y1).filter(|&y| labels[y * w + x] == id);
        if let Some(from) = hits.next() {
            let to = hits.last().unwrap_or(from);
            points.push([x as f64, from as f64]);
            if to != from {
                points.push([x as f64, to as f64]);
            }
        }
    }
    points
}

/// Hull, largest quadrilateral, then the sides fitted to the points again.
fn quad_from_points(points: &[Point]) -> Option<Quad> {
    if points.len() < MIN_OUTLINE {
        return None;
    }
    let corners = max_area_quad(&simplify_hull(convex_hull(points), HULL_KEEP))?;
    let quad = order_quad(refine_quad(order_quad(corners), points));
    plausible_quad(&quad).then_some(quad)
}

/// The outline of the patch with only the pixels that lie in a long run, so
/// nothing but its lines is left. A drawing that touches the frame drops out,
/// and with it the corner it would pull away.
fn lattice_points(labels: &[i32], w: usize, patch: &Patch) -> Vec<Point> {
    let (bw, bh) = (patch.width(), patch.height());
    let need = js::round_i(bw.max(bh) as f64 * RUN_SHARE) as usize;
    let keep = long_runs(&patch_mask(labels, w, patch), bw, bh, need, need);
    let whole = Patch { id: 1, count: 0, x0: 0, y0: 0, x1: bw - 1, y1: bh - 1 };
    let mut points = outline_points(&keep, bw, &whole, 1u8);
    for p in &mut points {
        p[0] += patch.x0 as f64;
        p[1] += patch.y0 as f64;
    }
    points
}

fn grid_likeness(gray: &[u8], w: usize, h: usize, quad: &Quad) -> f64 {
    let probe = SCORE_CELL * 9;
    let flat = rectify(gray, w, h, quad, probe);
    band_score(&threshold(&flat, probe, probe, SCORE_CELL / 2, INK_BIAS, 0.0), probe)
}

/// Picks the ink patch that reads as a Sudoku. Candidates are wide, roughly
/// square outlines rather than solid areas; the largest goes first, but the
/// line structure decides.
pub fn find_grid(gray: &[u8], ink: &[u8], w: usize, h: usize) -> Option<Quad> {
    let (labels, patches) = components(ink, w, h);
    let min_area = (w * h) as f64 * PATCH_AREA;
    let mut candidates: Vec<(&Patch, usize)> = patches
        .iter()
        .filter_map(|p| {
            let (bw, bh) = (p.width() as f64, p.height() as f64);
            let area = bw * bh;
            let fill = p.count as f64 / area;
            let fits = area >= min_area
                && bw / bh >= 1.0 / PATCH_RATIO
                && bw / bh <= PATCH_RATIO
                && (FILL_MIN..=FILL_MAX).contains(&fill);
            fits.then_some((p, area as usize))
        })
        .collect();
    candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

    let mut best: Option<(Quad, f64, &Patch)> = None;
    for &(patch, _) in candidates.iter().take(CANDIDATES) {
        let Some(quad) = quad_from_points(&outline_points(&labels, w, patch, patch.id)) else { continue };
        let score = grid_likeness(gray, w, h, &quad);
        if best.is_none_or(|b| score > b.1) {
            best = Some((quad, score, patch));
        }
        if score > GOOD_ENOUGH {
            break;
        }
    }
    let (quad, score, patch) = best?;

    // So little line structure means something is stuck to the grid.
    if score < WEAK_GRID
        && let Some(lattice) = quad_from_points(&lattice_points(&labels, w, patch))
        && grid_likeness(gray, w, h, &lattice) > score
    {
        return Some(lattice);
    }
    Some(quad)
}
