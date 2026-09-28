//! Finds a Sudoku grid in a photo, straightens it, cuts it into 81
//! normalised cells, reads each cell with a small neural network and lets
//! the solver settle the cells the network is unsure of.
//!
//! The arithmetic follows web/camera.js step for step, including the rounding
//! of Math.round, Math.hypot and Uint8ClampedArray, so both give the same
//! planes and cells for the same pixels.

mod cells;
mod detect;
mod geometry;
mod ink;
mod js;
mod lines;
pub mod net;
mod read;
mod reading;

pub use cells::{Cell, NORM, SHAPE, cut_cells};
pub use geometry::{Point, Quad};
pub use read::{Outcome, Puzzle, Reading, SURE, read_frame, read_puzzle};
pub use reading::{INK_BIAS, INK_TRIES, Plane, straighten};
