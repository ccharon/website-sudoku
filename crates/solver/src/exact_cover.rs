/*
 * 729 candidate rows of the form "digit d+1 sits in cell (r,c)" and 324
 * constraint columns in four groups of 81 each:
 *
 *   cell      0..80    r*9+c          cell (r,c) is filled
 *   row      81..161   81 + r*9 + d   digit d+1 occurs in row r
 *   column  162..242   162 + c*9 + d  digit d+1 occurs in column c
 *   box     243..323   243 + b*9 + d  digit d+1 occurs in box b
 */

use crate::grid::{CELLS, SIZE, box_of};

pub const NUM_ROWS: usize = CELLS * SIZE;
pub const NUM_COLS: usize = 4 * CELLS;
const COL_CELL: usize = 0;
const COL_ROW: usize = CELLS;
const COL_COL: usize = 2 * CELLS;
const COL_BOX: usize = 3 * CELLS;

pub fn row_index(row: usize, col: usize, digit: usize) -> usize {
    (row * SIZE + col) * SIZE + digit
}

/* A candidate packs the cell and the digit, the digit in the lower place. */
pub fn cell_of_candidate(candidate: usize) -> usize {
    candidate / SIZE
}

pub fn digit_of_candidate(candidate: usize) -> usize {
    candidate % SIZE
}

pub fn cols_of_row(candidate: usize) -> [usize; 4] {
    let digit = digit_of_candidate(candidate);
    let cell = cell_of_candidate(candidate);
    let (row, col) = (cell / SIZE, cell % SIZE);
    [
        COL_CELL + row * SIZE + col,
        COL_ROW + row * SIZE + digit,
        COL_COL + col * SIZE + digit,
        COL_BOX + box_of(row, col) * SIZE + digit,
    ]
}
