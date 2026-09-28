use crate::dlx::Solver;
use crate::grid::{CELLS, Grid, empty_grid};

/// Random numbers from a seed, so the same puzzle can be repeated. The
/// sequence matches randomFromSeed in the JavaScript version bit for bit.
pub struct Random {
    state: u32,
}

impl Random {
    pub fn from_seed(seed: u32) -> Self {
        Random { state: seed }
    }

    /// A number in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let a = self.state;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}

pub fn shuffle<T>(items: &mut [T], random: &mut Random) {
    for i in (1..items.len()).rev() {
        let j = (random.next_f64() * (i + 1) as f64) as usize;
        items.swap(i, j);
    }
}

impl Solver {
    /// Returns a random complete assignment.
    pub fn full_grid(&mut self, random: &mut Random) -> Grid {
        self.solve(&empty_grid(), Some(random))
            .expect("the empty grid has no conflict")
            .expect("the empty grid has a solution")
    }

    /// Returns a puzzle with exactly one solution. It starts from a complete
    /// assignment and keeps removing cells as long as the solution stays
    /// unique. With symmetric the cells are removed in point-symmetric pairs.
    pub fn generate(&mut self, seed: u32, symmetric: bool) -> Grid {
        let mut random = Random::from_seed(seed);
        let mut grid = self.full_grid(&mut random);

        let mut order: Vec<usize> = (0..CELLS).collect();
        shuffle(&mut order, &mut random);
        for i in order {
            let mirror = CELLS - 1 - i;
            let cells: &[usize] = if symmetric && i != mirror { &[i, mirror] } else { &[i] };

            // A pair is visited twice, and the second time nothing is left in it.
            let saved: Vec<u8> = cells.iter().map(|&c| grid[c]).collect();
            if saved.iter().all(|&v| v == 0) {
                continue;
            }
            for &c in cells {
                grid[c] = 0;
            }

            if self.count_solutions(&grid, 2) != Ok(1) {
                for (&c, &v) in cells.iter().zip(&saved) {
                    grid[c] = v;
                }
            }
        }
        grid
    }
}
