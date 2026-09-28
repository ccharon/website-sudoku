//! The solver as a WebAssembly module. A grid crosses the boundary as a
//! Uint8Array of 81 cells, 0 for empty.

use std::cell::RefCell;

use sudoku_solver::{CELLS, Grid, Solver};
use wasm_bindgen::prelude::*;

thread_local! {
    // Building the matrix once serves every call.
    static SOLVER: RefCell<Solver> = RefCell::new(Solver::new());
}

fn with_solver<T>(f: impl FnOnce(&mut Solver) -> T) -> T {
    SOLVER.with_borrow_mut(f)
}

fn to_grid(cells: &[u8]) -> Result<Grid, JsError> {
    cells
        .try_into()
        .map_err(|_| JsError::new(&format!("expected {CELLS} cells, got {}", cells.len())))
}

#[wasm_bindgen]
pub struct Conflict {
    index: usize,
    other_index: usize,
    text: String,
}

#[wasm_bindgen]
impl Conflict {
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> usize {
        self.index
    }

    #[wasm_bindgen(getter = otherIndex)]
    pub fn other_index(&self) -> usize {
        self.other_index
    }

    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
}

impl From<sudoku_solver::Conflict> for Conflict {
    fn from(c: sudoku_solver::Conflict) -> Self {
        Conflict { index: c.index, other_index: c.other_index, text: c.to_string() }
    }
}

#[wasm_bindgen]
pub fn validate(cells: &[u8]) -> Result<Vec<Conflict>, JsError> {
    let grid = to_grid(cells)?;
    Ok(sudoku_solver::validate(&grid).into_iter().map(Conflict::from).collect())
}

/// Returns the first solution found, or undefined. Throws the first conflict.
#[wasm_bindgen]
pub fn solve(cells: &[u8]) -> Result<Option<Vec<u8>>, JsError> {
    let grid = to_grid(cells)?;
    let solution = with_solver(|s| s.solve(&grid, None)).map_err(|c| JsError::new(&c.to_string()))?;
    Ok(solution.map(Vec::from))
}

#[wasm_bindgen(js_name = countSolutions)]
pub fn count_solutions(cells: &[u8], limit: usize) -> Result<usize, JsError> {
    let grid = to_grid(cells)?;
    with_solver(|s| s.count_solutions(&grid, limit)).map_err(|c| JsError::new(&c.to_string()))
}

#[wasm_bindgen]
pub fn generate(seed: u32, symmetric: bool) -> Vec<u8> {
    with_solver(|s| s.generate(seed, symmetric)).to_vec()
}

/// A straightened grid, one of the candidates `straighten` returns.
#[wasm_bindgen]
pub struct Plane(sudoku_ocr::Plane);

#[wasm_bindgen]
impl Plane {
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.0.size
    }

    /// The straightened grey plane, size × size.
    #[wasm_bindgen(getter)]
    pub fn flat(&self) -> Vec<u8> {
        self.0.flat.clone()
    }

    /// The corners in the photo as x0, y0, … x3, y3, top left first, clockwise.
    #[wasm_bindgen(getter)]
    pub fn quad(&self) -> Vec<f64> {
        self.0.quad.iter().flatten().copied().collect()
    }

    /// The 81 cells at one ink threshold.
    pub fn cells(&self, bias: f64) -> Cells {
        let cells = sudoku_ocr::cut_cells(&self.0.ink(bias), self.0.size);
        let mut out = Cells {
            present: vec![0; CELLS],
            faint: vec![0; CELLS],
            masks: vec![0; CELLS * MASK],
            shapes: vec![0.0; CELLS * SHAPE],
        };
        for (i, cell) in cells.iter().enumerate() {
            if let Some(cell) = cell {
                out.present[i] = 1;
                out.faint[i] = cell.faint as u8;
                out.masks[i * MASK..(i + 1) * MASK].copy_from_slice(&cell.mask);
                out.shapes[i * SHAPE..(i + 1) * SHAPE].copy_from_slice(&cell.shape);
            }
        }
        out
    }
}

const MASK: usize = sudoku_ocr::NORM * sudoku_ocr::NORM;
const SHAPE: usize = sudoku_ocr::SHAPE;

/// 81 cells in row order. A cell without a digit has present 0 and an empty mask.
#[wasm_bindgen(getter_with_clone)]
pub struct Cells {
    pub present: Vec<u8>,
    pub faint: Vec<u8>,
    /// 32 × 32 bytes per cell, one after the other.
    pub masks: Vec<u8>,
    /// Five shape measures per cell, as sudoku_ocr::Cell describes them.
    pub shapes: Vec<f32>,
}

/// Finds the grid in RGBA pixels and straightens it. Returns the candidate
/// planes, best first, or none if no grid was found.
#[wasm_bindgen]
pub fn straighten(rgba: &[u8], width: usize, height: usize, cell_px: Option<usize>, dark: bool) -> Result<Vec<Plane>, JsError> {
    check_pixels(rgba, width, height)?;
    let planes = sudoku_ocr::straighten(rgba, width, height, cell_px, dark).unwrap_or_default();
    Ok(planes.into_iter().map(Plane).collect())
}

static DIGITS: &[u8] = include_bytes!("../../../models/digits.bin");

thread_local! {
    static NET: sudoku_ocr::net::Net = sudoku_ocr::net::Net::from_le_bytes(DIGITS).expect("models/digits.bin fits the network");
}

/// Class probabilities for masks of 32 × 32 bytes each with five shape
/// measures each, as Cells holds them. Ten per cell: no digit first, then
/// the digits 1 to 9.
#[wasm_bindgen]
pub fn classify(masks: &[u8], shapes: &[f32]) -> Result<Vec<f32>, JsError> {
    if !masks.len().is_multiple_of(MASK) || masks.len() / MASK * SHAPE != shapes.len() {
        return Err(JsError::new(&format!("{} mask bytes and {} shape measures do not fit together", masks.len(), shapes.len())));
    }
    Ok(NET.with(|net| {
        masks
            .chunks(MASK)
            .zip(shapes.as_chunks::<SHAPE>().0)
            .flat_map(|(m, s)| net.classify(m, s))
            .collect()
    }))
}

/// What reading a photo gave. With ok the grid is the puzzle, else reason
/// says what went wrong. readingGrid is the net's first choice per cell, if
/// the digits were read at all.
#[wasm_bindgen(getter_with_clone)]
pub struct PuzzleRead {
    pub ok: bool,
    pub grid: Vec<u8>,
    pub reason: Option<String>,
    pub changed: Vec<u32>,
    pub uncertain: Vec<u8>,
    #[wasm_bindgen(js_name = readingGrid)]
    pub reading_grid: Option<Vec<u8>>,
    pub quad: Option<Vec<f64>>,
}

impl From<sudoku_ocr::Outcome> for PuzzleRead {
    fn from(outcome: sudoku_ocr::Outcome) -> Self {
        let quad = |r: &sudoku_ocr::Reading| r.quad.iter().flatten().copied().collect();
        match outcome {
            sudoku_ocr::Outcome::Solved(p) => PuzzleRead {
                ok: true,
                grid: p.grid.to_vec(),
                reason: None,
                changed: p.changed.iter().map(|&i| i as u32).collect(),
                uncertain: p.uncertain.iter().map(|&u| u as u8).collect(),
                quad: Some(quad(&p.reading)),
                reading_grid: Some(p.reading.grid.to_vec()),
            },
            sudoku_ocr::Outcome::Failed { reason, reading } => PuzzleRead {
                ok: false,
                grid: Vec::new(),
                reason: Some(reason.to_string()),
                changed: Vec::new(),
                uncertain: Vec::new(),
                quad: reading.as_ref().map(quad),
                reading_grid: reading.map(|r| r.grid.to_vec()),
            },
        }
    }
}

fn check_pixels(rgba: &[u8], width: usize, height: usize) -> Result<(), JsError> {
    if rgba.len() != width * height * 4 {
        return Err(JsError::new(&format!("expected {} bytes for {width}x{height}, got {}", width * height * 4, rgba.len())));
    }
    Ok(())
}

/// Reads a photo, dark ink first and reversed after.
#[wasm_bindgen(js_name = readPuzzle)]
pub fn read_puzzle(rgba: &[u8], width: usize, height: usize, cell_px: Option<usize>) -> Result<PuzzleRead, JsError> {
    check_pixels(rgba, width, height)?;
    Ok(NET.with(|net| with_solver(|s| sudoku_ocr::read_puzzle(net, s, rgba, width, height, cell_px))).into())
}

/// Reads one frame at one polarity.
#[wasm_bindgen(js_name = readFrame)]
pub fn read_frame(rgba: &[u8], width: usize, height: usize, cell_px: Option<usize>, dark: bool) -> Result<PuzzleRead, JsError> {
    check_pixels(rgba, width, height)?;
    Ok(NET.with(|net| with_solver(|s| sudoku_ocr::read_frame(net, s, rgba, width, height, cell_px, dark))).into())
}
