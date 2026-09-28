//! Checks the solver against the puzzles in ../sudoku-solver/solver/testdata,
//! or the directory named by SUDOKU_TESTDATA.

use std::fs;
use std::path::PathBuf;

use sudoku_solver::{CELLS, ConflictKind, Grid, Solver, Unit, grid_to_string, is_complete, parse_grid, validate};

fn testdata() -> PathBuf {
    std::env::var_os("SUDOKU_TESTDATA").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../sudoku-solver/solver/testdata")
    })
}

/// Pairs each puzzle with the comment line above it, if there is one.
fn cases(file: &str) -> Vec<(String, Grid)> {
    let path = testdata().join(file);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut name = String::new();
    let mut out = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Some(comment) = line.strip_prefix('#') {
            name = comment.trim().to_string();
            continue;
        }
        let grid = parse_grid(line).unwrap_or_else(|e| panic!("{file}: {e}: {line}"));
        out.push((std::mem::take(&mut name), grid));
    }
    out
}

fn check_solution(puzzle: &Grid, solution: &Grid) {
    let p = grid_to_string(puzzle);
    assert!(is_complete(solution), "incomplete for {p}");
    assert!(validate(solution).is_empty(), "conflicting for {p}");
    for i in 0..CELLS {
        assert!(puzzle[i] == 0 || puzzle[i] == solution[i], "given {i} changed for {p}");
    }
}

#[test]
fn every_puzzle_has_one_checked_solution() {
    let mut solver = Solver::new();
    for file in ["euler96.txt", "hardest.txt", "top95.txt"] {
        let puzzles = cases(file);
        assert!(!puzzles.is_empty(), "{file} is empty");
        for (_, puzzle) in puzzles {
            let solution = solver.solve(&puzzle, None).unwrap().expect("a solution");
            check_solution(&puzzle, &solution);
            assert_eq!(solver.count_solutions(&puzzle, 2), Ok(1), "{}", grid_to_string(&puzzle));
        }
    }
}

#[test]
fn edge_cases() {
    let cases = cases("edgecases.txt");
    assert_eq!(cases.len(), 7);
    let mut solver = Solver::new();
    let twice = |unit| Some(ConflictKind::Twice(unit));
    let expected = [
        (None, 3),
        (None, 1),
        (twice(Unit::Row), 0),
        (twice(Unit::Column), 0),
        (twice(Unit::Box), 0),
        (None, 0),
        (None, 3),
    ];
    for ((name, puzzle), (conflict, count)) in cases.iter().zip(expected) {
        let first = validate(puzzle).first().map(|c| c.kind);
        assert_eq!(first, conflict, "{name}");
        match conflict {
            Some(_) => assert!(solver.solve(puzzle, None).is_err(), "{name}"),
            None => {
                assert_eq!(solver.count_solutions(puzzle, 3), Ok(count), "{name}");
                match solver.solve(puzzle, None).unwrap() {
                    Some(solution) => check_solution(puzzle, &solution),
                    None => assert_eq!(count, 0, "{name}"),
                }
            }
        }
    }
}

/// Fixed puzzles per seed, so a change to the random sequence or the search
/// order shows up.
#[test]
fn generate_repeats_per_seed() {
    let mut solver = Solver::new();
    let expected = [
        (1, true, "8.4...5..6..41..2..723.8......785.3...........8.296......5.386..4..67..1..8...3.2"),
        (42, true, "85.....9..41..3.566.....47.7...24......1.8......73...8.73.....918.3..76..6.....21"),
        (u32::MAX, true, ".5..1...9..6..3....7.9.4.8.1..7.....937...261.....1..8.8.3.2.9....8..5..3...7..2."),
        (7, false, "6....8...278..61...4.....9.83.4..7...2..5...3.....7.6.......4.1...7..6.........82"),
    ];
    for (seed, symmetric, puzzle) in expected {
        let grid = solver.generate(seed, symmetric);
        assert_eq!(grid_to_string(&grid), puzzle, "seed {seed}");
        assert_eq!(solver.count_solutions(&grid, 2), Ok(1));
    }
}
