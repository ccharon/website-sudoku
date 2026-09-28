//! Reads a puzzle off a photo: the net gives every cell a probability per
//! class, and the solver picks the most probable reading that is free of
//! conflicts and has exactly one solution.

use sudoku_solver::{CELLS, Grid, Solver, validate};

use crate::cells::cut_cells;
use crate::geometry::Quad;
use crate::net::{CLASSES, Net};
use crate::reading::{INK_TRIES, Plane, straighten};

/// Below this probability a cell is shown as unconfirmed.
pub const SURE: f32 = 0.9;

const MAX_CHANGES: usize = 3; // cells the search may read differently from the net's first choice
const MAX_COST: f64 = 12.0; // log likelihood a reading may lose against the net's first choice
const ALTERNATIVES: usize = 40; // cheapest single changes the search combines
const RESTORE_FLOOR: f32 = 0.05; // chance the net must give a digit before it is put back

/// What the net made of the cells of one plane at one threshold.
#[derive(Debug, Clone)]
pub struct Reading {
    pub quad: Quad,
    /// The first choice per cell, 0 where it saw no digit.
    pub grid: Grid,
    /// Class probabilities per cell, None where the cell held no ink.
    pub probs: Vec<Option<[f32; CLASSES]>>,
    /// Cells with too little ink to call a digit.
    pub faint: [bool; CELLS],
}

#[derive(Debug, Clone)]
pub struct Puzzle {
    pub grid: Grid,
    /// Cells read differently from the net's first choice.
    pub changed: Vec<usize>,
    /// Cells changed, or whose digit the net was not sure of.
    pub uncertain: [bool; CELLS],
    /// Log likelihood the grid lost against the net's first choice.
    pub cost: f64,
    pub reading: Reading,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Solved(Puzzle),
    Failed { reason: &'static str, reading: Option<Reading> },
}

impl Outcome {
    fn reading(&self) -> Option<&Reading> {
        match self {
            Outcome::Solved(p) => Some(&p.reading),
            Outcome::Failed { reading, .. } => reading.as_ref(),
        }
    }
}

fn read_cells(net: &Net, plane: &Plane, bias: f64) -> Reading {
    let cells = cut_cells(&plane.ink(bias), plane.size);
    let probs: Vec<Option<[f32; CLASSES]>> = cells.iter().map(|c| c.as_ref().map(|c| net.classify(&c.mask, &c.shape))).collect();
    let faint = std::array::from_fn(|i| cells[i].as_ref().is_some_and(|c| c.faint));
    let mut grid = [0; CELLS];
    for (g, p) in grid.iter_mut().zip(&probs) {
        if let Some(p) = p {
            *g = (0..CLASSES).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0) as u8;
        }
    }
    Reading { quad: plane.quad, grid, probs, faint }
}

/// The cheapest set of up to MAX_CHANGES changes to the first choices that
/// leaves a grid without conflicts and with exactly one solution.
fn resolve(solver: &mut Solver, reading: Reading) -> Option<Puzzle> {
    let mut alternatives = Vec::new();
    for (i, p) in reading.probs.iter().enumerate() {
        let Some(p) = p else { continue };
        let best = (p[reading.grid[i] as usize] as f64).ln();
        for (k, &pk) in p.iter().enumerate() {
            let cost = best - (pk as f64).max(1e-9).ln();
            if k != reading.grid[i] as usize && cost < MAX_COST {
                alternatives.push((cost, i, k as u8));
            }
        }
    }
    alternatives.sort_by(|a, b| a.0.total_cmp(&b.0));
    alternatives.truncate(ALTERNATIVES);

    // Every combination of up to MAX_CHANGES alternatives in distinct cells, cheapest first.
    let mut combos: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
    let mut frontier = combos.clone();
    for _ in 0..MAX_CHANGES {
        let mut next = Vec::new();
        for (cost, picked) in &frontier {
            let from = picked.last().map_or(0, |&l| l + 1);
            for (j, alt) in alternatives.iter().enumerate().skip(from) {
                let total = cost + alt.0;
                if total < MAX_COST && picked.iter().all(|&q| alternatives[q].1 != alt.1) {
                    let mut more = picked.clone();
                    more.push(j);
                    next.push((total, more));
                }
            }
        }
        combos.extend(next.iter().cloned());
        frontier = next;
    }
    combos.sort_by(|a, b| a.0.total_cmp(&b.0));

    for (cost, picked) in combos {
        let mut grid = reading.grid;
        for &j in &picked {
            grid[alternatives[j].1] = alternatives[j].2;
        }
        if !validate(&grid).is_empty() || solver.count_solutions(&grid, 2) != Ok(1) {
            continue;
        }
        let mut changed: Vec<usize> = picked.iter().map(|&j| alternatives[j].1).collect();
        restore_givens(solver, &reading, &mut grid, &mut changed);
        let uncertain = std::array::from_fn(|i| {
            changed.contains(&i) || reading.probs[i].is_some_and(|p| p[grid[i] as usize] < SURE)
        });
        return Some(Puzzle { grid, changed, uncertain, cost, reading });
    }
    None
}

/// Puts back givens the net took for empty cells. The net is too sure of
/// "no digit" on some faces, but a cell with plenty of ink in its middle is
/// seldom empty. Once the solution is known, such a cell gets the digit the
/// solution has there, if that is also the net's most likely digit for it
/// and the net gives it a real chance.
fn restore_givens(solver: &mut Solver, reading: &Reading, grid: &mut Grid, changed: &mut Vec<usize>) {
    let Ok(Some(solution)) = solver.solve(grid, None) else { return };
    for i in 0..CELLS {
        let Some(p) = reading.probs[i] else { continue };
        if grid[i] != 0 || reading.faint[i] {
            continue;
        }
        let digit = (1..CLASSES).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0) as u8;
        if digit == solution[i] && p[digit as usize] >= RESTORE_FLOOR {
            grid[i] = digit;
            changed.push(i);
        }
    }
}

/// Reads one frame at one polarity. Every candidate plane is read at every
/// threshold, and the reading that needs the fewest changes wins. One that
/// needs none ends the search at once.
pub fn read_frame(
    net: &Net,
    solver: &mut Solver,
    rgba: &[u8],
    w: usize,
    h: usize,
    cell_px: Option<usize>,
    dark: bool,
) -> Outcome {
    let Some(planes) = straighten(rgba, w, h, cell_px, dark) else {
        return Outcome::Failed { reason: "no grid found", reading: None };
    };
    let mut first = None;
    let mut best: Option<Puzzle> = None;
    for plane in &planes {
        for bias in INK_TRIES {
            let reading = read_cells(net, plane, bias);
            first.get_or_insert_with(|| reading.clone());
            let Some(puzzle) = resolve(solver, reading) else { continue };
            if puzzle.cost == 0.0 {
                return Outcome::Solved(puzzle);
            }
            if best.as_ref().is_none_or(|b| puzzle.cost < b.cost) {
                best = Some(puzzle);
            }
        }
    }
    match best {
        Some(p) => Outcome::Solved(p),
        None => Outcome::Failed { reason: "grid read, but it does not solve", reading: first },
    }
}

/// Reads one photo, dark ink on a light ground first. A screen in dark mode
/// turns that around, so the whole chain runs a second time reversed.
pub fn read_puzzle(net: &Net, solver: &mut Solver, rgba: &[u8], w: usize, h: usize, cell_px: Option<usize>) -> Outcome {
    let mut last = None;
    for dark in [false, true] {
        let outcome = read_frame(net, solver, rgba, w, h, cell_px, dark);
        if let Outcome::Solved(_) = outcome {
            return outcome;
        }
        // The first reading that got as far as the digits explains the failure best.
        if last.is_none() && outcome.reading().is_some() {
            last = Some(outcome);
        }
    }
    last.unwrap_or(Outcome::Failed { reason: "no grid found", reading: None })
}
