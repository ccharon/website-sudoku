//! Sudoku as an exact cover problem, solved with Knuth's Algorithm X as
//! Dancing Links.

mod dlx;
mod exact_cover;
mod generator;
mod grid;

pub use dlx::Solver;
pub use generator::Random;
pub use grid::{
    BOX_SIZE, CELLS, Conflict, ConflictKind, Grid, ParseError, SIZE, Unit, empty_grid, givens,
    grid_to_string, is_complete, parse_grid, validate,
};
