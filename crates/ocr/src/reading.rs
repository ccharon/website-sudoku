use crate::detect::find_grid;
use crate::geometry::{Quad, rect_quad, rectify, to_photo};
use crate::ink::{invert_plane, threshold, to_gray};
use crate::js;
use crate::lines::{band_score, grid_bounds, paper_quad};

pub const INK_BIAS: f64 = 0.9; // darker than this share of the local mean counts as ink
const INK_FIND: f64 = 0.88; // threshold of the pass that looks for the outline
const FIND_STEP: f64 = 10.0; // grey levels the outline pass needs below the mean
pub const INK_TRIES: [f64; 3] = [0.9, 0.96, 0.84]; // retried in this order until the reading solves

const WARP_STILL: usize = 80; // pixels per cell for a single picture
const CUT_GAIN: f64 = 0.05; // how much more regular the lines must get for a cut to count
const CUT_SURE: f64 = 0.5; // line score a cut must reach when the uncut grid scored nothing
const FIND_CELLS: f64 = 24.0; // the outline pass assumes a photo about this many cells wide
const FIND_RADIUS: isize = 4; // and never averages over fewer pixels than this

/// A grey plane straightened along one quadrilateral of the photo.
#[derive(Debug, Clone)]
pub struct Plane {
    pub flat: Vec<u8>,
    pub size: usize,
    pub radius: usize,
    pub quad: Quad,
}

impl Plane {
    fn new(gray: &[u8], w: usize, h: usize, quad: Quad, cell: usize) -> Self {
        let size = cell * 9;
        Plane {
            flat: rectify(gray, w, h, &quad, size),
            size,
            radius: js::round_i(cell as f64 / 2.0) as usize,
            quad,
        }
    }

    /// The ink mask of the plane at one threshold.
    pub fn ink(&self, bias: f64) -> Vec<u8> {
        threshold(&self.flat, self.size, self.size, self.radius, bias, 0.0)
    }
}

/// The corners of the grid in a grey plane, or None.
fn locate(gray: &[u8], w: usize, h: usize) -> Option<Quad> {
    let radius = FIND_RADIUS.max(js::round_i(w.min(h) as f64 / FIND_CELLS)) as usize;
    find_grid(gray, &threshold(gray, w, h, radius, INK_FIND, FIND_STEP), w, h)
}

/// Cuts the plane down to the 9x9 grid when a title box shares the outline.
/// The cut is kept only if the lines come out more regular for it. A grid that
/// scored nothing gives any cut that edge, so such a cut must score well itself.
fn drop_title(gray: &[u8], w: usize, h: usize, plane: Plane, cell: usize) -> Plane {
    let before = plane.ink(INK_BIAS);
    let Some(bounds) = grid_bounds(&before, plane.size) else { return plane };
    let cut = Plane::new(gray, w, h, to_photo(&plane.quad, &rect_quad(&bounds), plane.size), cell);
    let was = band_score(&before, plane.size);
    let now = band_score(&cut.ink(INK_BIAS), cut.size);
    if now > was + CUT_GAIN && (was > 0.0 || now > CUT_SURE) { cut } else { plane }
}

/// The ways to straighten a plane whose outline took in a dark surround, best
/// first. The line rating cannot choose among them, so the solver does. With
/// the surround cut off, a grid it had merged with stands free and is looked
/// for again, and the uncut plane stays as the last resort.
fn surround_cuts(gray: &[u8], w: usize, h: usize, plane: Plane, cell: usize) -> Vec<Plane> {
    let Some(paper) = paper_quad(&plane.flat, plane.size) else { return vec![plane] };
    let out = Plane::new(gray, w, h, to_photo(&plane.quad, &paper, plane.size), cell);
    let Some(inner) = locate(&out.flat, out.size, out.size) else { return vec![out, plane] };
    let mut free = Plane::new(gray, w, h, to_photo(&out.quad, &inner, out.size), cell);
    if let Some(frame) = paper_quad(&free.flat, free.size) {
        free = Plane::new(gray, w, h, to_photo(&free.quad, &frame, free.size), cell);
    }
    vec![out, free, plane]
}

/// Finds the grid in RGBA pixels and straightens it, the costly half of a
/// reading. With dark set the grey plane is reversed first, for light ink on
/// a dark ground. Returns the candidate planes, best first, or None.
pub fn straighten(rgba: &[u8], w: usize, h: usize, cell_px: Option<usize>, dark: bool) -> Option<Vec<Plane>> {
    let gray = to_gray(rgba);
    let gray = if dark { invert_plane(&gray) } else { gray };
    let quad = locate(&gray, w, h)?;
    let cell = cell_px.unwrap_or(WARP_STILL);
    let planes = surround_cuts(&gray, w, h, Plane::new(&gray, w, h, quad, cell), cell);
    Some(planes.into_iter().map(|p| drop_title(&gray, w, h, p, cell)).collect())
}
