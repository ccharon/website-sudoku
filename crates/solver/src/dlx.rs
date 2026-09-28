/*
 * Node layout: index 0 is the root, 1 to 324 are the column heads, then four
 * data nodes per candidate row.
 */

use crate::exact_cover::{NUM_COLS, NUM_ROWS, cell_of_candidate, cols_of_row, digit_of_candidate, row_index};
use crate::generator::{Random, shuffle};
use crate::grid::{CELLS, Conflict, Grid, SIZE, empty_grid, validate};

const ROOT: usize = 0;
const NODE_BASE: usize = 1 + NUM_COLS;
const NODES_PER_ROW: usize = 4;
const TOTAL_NODES: usize = NODE_BASE + NUM_ROWS * NODES_PER_ROW;

fn head_of(col: usize) -> usize {
    col + 1
}

/// Holds the linked matrix, which is the same for every Sudoku. Each run
/// leaves it as it found it, so one solver serves any number of puzzles.
pub struct Solver {
    left: Vec<usize>,
    right: Vec<usize>,
    up: Vec<usize>,
    down: Vec<usize>,
    col: Vec<usize>,
    row: Vec<usize>,
    size: Vec<usize>,
    sol: Vec<usize>,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    pub fn new() -> Self {
        let mut m = Solver {
            left: vec![0; TOTAL_NODES],
            right: vec![0; TOTAL_NODES],
            up: vec![0; TOTAL_NODES],
            down: vec![0; TOTAL_NODES],
            col: vec![0; TOTAL_NODES],
            row: vec![0; TOTAL_NODES],
            size: vec![0; NODE_BASE],
            sol: Vec::with_capacity(CELLS),
        };

        let mut prev = ROOT;
        for c in 0..NUM_COLS {
            let h = head_of(c);
            m.left[h] = prev;
            m.right[prev] = h;
            m.up[h] = h;
            m.down[h] = h;
            m.col[h] = h;
            prev = h;
        }
        m.right[prev] = ROOT;
        m.left[ROOT] = prev;

        for r in 0..NUM_ROWS {
            let base = NODE_BASE + r * NODES_PER_ROW;
            for (k, c) in cols_of_row(r).into_iter().enumerate() {
                let n = base + k;
                m.row[n] = r;
                m.left[n] = base + (k + NODES_PER_ROW - 1) % NODES_PER_ROW;
                m.right[n] = base + (k + 1) % NODES_PER_ROW;

                let h = head_of(c);
                let last = m.up[h];
                m.down[last] = n;
                m.up[n] = last;
                m.down[n] = h;
                m.up[h] = n;
                m.col[n] = h;
                m.size[h] += 1;
            }
        }
        m
    }

    /// Removes the column from the header row and every row that fills it.
    fn cover(&mut self, h: usize) {
        self.right[self.left[h]] = self.right[h];
        self.left[self.right[h]] = self.left[h];
        let mut i = self.down[h];
        while i != h {
            let mut j = self.right[i];
            while j != i {
                self.down[self.up[j]] = self.down[j];
                self.up[self.down[j]] = self.up[j];
                self.size[self.col[j]] -= 1;
                j = self.right[j];
            }
            i = self.down[i];
        }
    }

    /// Undoes cover. The reverse order restores the state exactly, because
    /// every removed node kept its neighbours.
    fn uncover(&mut self, h: usize) {
        let mut i = self.up[h];
        while i != h {
            let mut j = self.left[i];
            while j != i {
                self.size[self.col[j]] += 1;
                self.down[self.up[j]] = j;
                self.up[self.down[j]] = j;
                j = self.left[j];
            }
            i = self.up[i];
        }
        self.right[self.left[h]] = h;
        self.left[self.right[h]] = h;
    }

    /// Returns the head node of the constraint with the fewest remaining
    /// options, or None once every constraint is satisfied.
    fn choose_column(&self) -> Option<usize> {
        let mut best = None;
        let mut best_size = usize::MAX;
        let mut h = self.right[ROOT];
        while h != ROOT {
            if self.size[h] < best_size {
                best = Some(h);
                best_size = self.size[h];
                if best_size <= 1 {
                    break;
                }
            }
            h = self.right[h];
        }
        best
    }

    fn select_row(&mut self, r: usize) {
        let base = NODE_BASE + r * NODES_PER_ROW;
        for k in 0..NODES_PER_ROW {
            self.cover(self.col[base + k]);
        }
        self.sol.push(r);
    }

    fn unselect_row(&mut self, r: usize) {
        let base = NODE_BASE + r * NODES_PER_ROW;
        for k in (0..NODES_PER_ROW).rev() {
            self.uncover(self.col[base + k]);
        }
        self.sol.pop();
    }

    /// Searches for assignments that satisfy every remaining constraint
    /// exactly once. on_solution ends the search with true. The matrix is
    /// restored in any case before this returns.
    fn search(
        &mut self,
        on_solution: &mut dyn FnMut(&[usize]) -> bool,
        random: &mut Option<&mut Random>,
    ) -> bool {
        let Some(h) = self.choose_column() else {
            return on_solution(&self.sol);
        };
        if self.size[h] == 0 {
            return false;
        }

        self.cover(h);

        let mut rows = Vec::with_capacity(self.size[h]);
        let mut i = self.down[h];
        while i != h {
            rows.push(i);
            i = self.down[i];
        }
        if let Some(random) = random {
            shuffle(&mut rows, random);
        }

        for i in rows {
            self.sol.push(self.row[i]);
            let mut j = self.right[i];
            while j != i {
                self.cover(self.col[j]);
                j = self.right[j];
            }

            let stop = self.search(on_solution, random);

            let mut j = self.left[i];
            while j != i {
                self.uncover(self.col[j]);
                j = self.left[j];
            }
            self.sol.pop();
            if stop {
                self.uncover(h);
                return true;
            }
        }
        self.uncover(h);
        false
    }

    /// Applies the givens, searches and restores the matrix.
    fn run(
        &mut self,
        grid: &Grid,
        on_solution: &mut dyn FnMut(&[usize]) -> bool,
        mut random: Option<&mut Random>,
    ) -> Result<(), Conflict> {
        if let Some(&conflict) = validate(grid).first() {
            return Err(conflict);
        }
        let mut applied = 0;
        for (i, &value) in grid.iter().enumerate() {
            if value != 0 {
                self.select_row(row_index(i / SIZE, i % SIZE, value as usize - 1));
                applied += 1;
            }
        }
        self.search(on_solution, &mut random);
        // search leaves the solution as it found it, so the givens lie on top.
        for _ in 0..applied {
            let r = *self.sol.last().expect("a given is still selected");
            self.unselect_row(r);
        }
        Ok(())
    }

    /// Returns the first solution found, or None. The first conflict in the
    /// givens is the error.
    pub fn solve(&mut self, grid: &Grid, random: Option<&mut Random>) -> Result<Option<Grid>, Conflict> {
        let mut solution = None;
        self.run(
            grid,
            &mut |sol| {
                solution = Some(grid_from_rows(sol));
                true
            },
            random,
        )?;
        Ok(solution)
    }

    /// Counts the solutions and stops at limit. A limit of 2 is enough to
    /// check uniqueness.
    pub fn count_solutions(&mut self, grid: &Grid, limit: usize) -> Result<usize, Conflict> {
        let max = limit.max(1);
        let mut count = 0;
        self.run(
            grid,
            &mut |_| {
                count += 1;
                count >= max
            },
            None,
        )?;
        Ok(count)
    }
}

fn grid_from_rows(sol: &[usize]) -> Grid {
    let mut grid = empty_grid();
    for &candidate in sol {
        grid[cell_of_candidate(candidate)] = digit_of_candidate(candidate) as u8 + 1;
    }
    grid
}
