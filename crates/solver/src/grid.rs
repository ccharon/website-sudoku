use std::fmt;

pub const SIZE: usize = 9;
pub const BOX_SIZE: usize = 3;
pub const CELLS: usize = SIZE * SIZE;

/// 81 cells in row order, 0 for empty.
pub type Grid = [u8; CELLS];

pub fn empty_grid() -> Grid {
    [0; CELLS]
}

pub fn box_of(row: usize, col: usize) -> usize {
    row / BOX_SIZE * BOX_SIZE + col / BOX_SIZE
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    UnexpectedChar(char),
    TooManyCells,
    TooFewCells(usize),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::UnexpectedChar(ch) => write!(f, "unexpected character {ch:?}"),
            ParseError::TooManyCells => write!(f, "more than 81 cells"),
            ParseError::TooFewCells(n) => write!(f, "expected 81 cells, read {n}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Reads 81 cells. The characters '.', '0' and ' ' mean an empty cell, line
/// breaks and the frame characters '|', '-' and '+' are skipped.
pub fn parse_grid(text: &str) -> Result<Grid, ParseError> {
    let mut grid = empty_grid();
    let mut n = 0;
    for ch in text.chars() {
        let value = match ch {
            '1'..='9' => ch as u8 - b'0',
            '.' | '0' | ' ' => 0,
            '\n' | '\r' | '\t' | '|' | '-' | '+' => continue,
            _ => return Err(ParseError::UnexpectedChar(ch)),
        };
        if n == CELLS {
            return Err(ParseError::TooManyCells);
        }
        grid[n] = value;
        n += 1;
    }
    if n != CELLS {
        return Err(ParseError::TooFewCells(n));
    }
    Ok(grid)
}

pub fn grid_to_string(grid: &Grid) -> String {
    grid.iter()
        .map(|&v| if v == 0 { '.' } else { (b'0' + v) as char })
        .collect()
}

pub fn givens(grid: &Grid) -> usize {
    grid.iter().filter(|&&v| v != 0).count()
}

pub fn is_complete(grid: &Grid) -> bool {
    grid.iter().all(|&v| v != 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Row,
    Column,
    Box,
}

impl Unit {
    fn name(self) -> &'static str {
        match self {
            Unit::Row => "row",
            Unit::Column => "column",
            Unit::Box => "box",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictKind {
    /// The cell holds something other than a digit from 1 to 9.
    Value,
    /// The digit already stands at `other_index` in this unit.
    Twice(Unit),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conflict {
    pub index: usize,
    pub other_index: usize,
    pub value: u8,
    pub kind: ConflictKind,
}

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (row, col) = (self.index / SIZE + 1, self.index % SIZE + 1);
        match self.kind {
            ConflictKind::Value => {
                write!(f, "r{row}c{col} holds {}, not a digit from 1 to 9", self.value)
            }
            ConflictKind::Twice(unit) => write!(
                f,
                "Digit {} twice in {}: r{}c{}, r{row}c{col}",
                self.value,
                unit.name(),
                self.other_index / SIZE + 1,
                self.other_index % SIZE + 1,
            ),
        }
    }
}

/* A digit indexes its slot directly, so slot 0 of every unit stays unused. */
const DIGIT_SLOTS: usize = SIZE + 1;

/// Reports every cell outside the digit range and every unit that holds a
/// digit twice. Each affected cell is reported once, with the unit in the
/// order row, column, box.
pub fn validate(grid: &Grid) -> Vec<Conflict> {
    let mut row_first = [None; SIZE * DIGIT_SLOTS];
    let mut col_first = [None; SIZE * DIGIT_SLOTS];
    let mut box_first = [None; SIZE * DIGIT_SLOTS];
    let mut conflicts = Vec::new();

    for (i, &value) in grid.iter().enumerate() {
        if value == 0 {
            continue;
        }
        if value as usize > SIZE {
            conflicts.push(Conflict { index: i, other_index: i, value, kind: ConflictKind::Value });
            continue;
        }
        let (row, col) = (i / SIZE, i % SIZE);
        let d = value as usize;
        let ri = row * DIGIT_SLOTS + d;
        let ci = col * DIGIT_SLOTS + d;
        let bi = box_of(row, col) * DIGIT_SLOTS + d;

        let clash = row_first[ri]
            .map(|o| (o, Unit::Row))
            .or(col_first[ci].map(|o| (o, Unit::Column)))
            .or(box_first[bi].map(|o| (o, Unit::Box)));
        if let Some((other, unit)) = clash {
            conflicts.push(Conflict { index: i, other_index: other, value, kind: ConflictKind::Twice(unit) });
        }
        row_first[ri].get_or_insert(i);
        col_first[ci].get_or_insert(i);
        box_first[bi].get_or_insert(i);
    }
    conflicts
}
