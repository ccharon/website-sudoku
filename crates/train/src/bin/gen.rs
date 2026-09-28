//! Renders synthetic photos, runs them through sudoku-ocr and stores every
//! cell the pipeline cuts out with its true label.
//!
//!   gen <fonts dir> <out dir> <photos> [seed] [--val]
//!
//! Without --val the fonts come from all families except the held-out ones,
//! with --val only from those. With --show the first photos are also written
//! as PGM for a look.

use std::path::{Path, PathBuf};
use std::{env, fs, process, thread};

use sudoku_ocr::{INK_TRIES, Quad, cut_cells, straighten};
use sudoku_solver::Solver;
use sudoku_train::data;
use sudoku_train::render::{FAMILIES, Face, draw_sheet, load_families, photograph};
use sudoku_train::rng::Rng;

/// Families kept out of training, to measure how the net does on unseen fonts.
const HELD_OUT: [&str; 2] = ["URWGothic-", "NotoSerif-"];

/// How far a found corner may lie from the true one, in cells, besides half the frame.
const CORNER_SLACK: f64 = 0.25;

#[derive(Default)]
struct Tally {
    photos: usize,
    missed: usize,
    per_class: [usize; 10],
}

fn corner_error(a: &Quad, b: &Quad) -> f64 {
    (0..4).map(|i| (a[i][0] - b[i][0]).hypot(a[i][1] - b[i][1])).fold(0.0, f64::max)
}

fn cell_side(q: &Quad) -> f64 {
    (0..4).map(|i| (q[(i + 1) % 4][0] - q[i][0]).hypot(q[(i + 1) % 4][1] - q[i][1])).sum::<f64>() / 36.0
}

fn write_pgm(path: &Path, rgba: &[u8], w: usize, h: usize) {
    let mut out = format!("P5\n{w} {h}\n255\n").into_bytes();
    out.extend(rgba.chunks(4).map(|p| p[0]));
    fs::write(path, out).expect("write pgm");
}

fn one_photo(seed: u32, families: &[Vec<Face>], solver: &mut Solver, show: Option<&Path>, out: &mut Vec<u8>, tally: &mut Tally) {
    let mut rng = Rng::new(seed);
    let family = &families[rng.below(families.len())];
    let face = &family[rng.below(family.len())];
    let marks = &families[rng.below(families.len())];
    let marks = &marks[rng.below(marks.len())];
    let puzzle = solver.generate(seed.wrapping_mul(2654435761), rng.chance(0.7));

    let sheet = draw_sheet(&mut rng, face, marks, &puzzle);
    let shot = photograph(&mut rng, &sheet);
    if let Some(dir) = show {
        write_pgm(&dir.join(format!("photo-{seed}.pgm")), &shot.rgba, shot.w, shot.h);
    }
    tally.photos += 1;

    let cell_px = if rng.chance(0.5) { 64 } else { 80 };
    let planes = straighten(&shot.rgba, shot.w, shot.h, Some(cell_px), shot.dark).unwrap_or_default();
    // The found corners may sit on the outer edge of the frame.
    let slack = (CORNER_SLACK + sheet.frame / 2.0) * cell_side(&shot.grid);
    let Some(plane) = planes.iter().find(|p| corner_error(&p.quad, &shot.grid) < slack) else {
        tally.missed += 1;
        return;
    };
    for bias in INK_TRIES {
        for (i, cell) in cut_cells(&plane.ink(bias), plane.size).iter().enumerate() {
            if let Some(cell) = cell {
                data::push(out, puzzle[i], &cell.mask, &cell.shape);
                tally.per_class[puzzle[i] as usize] += 1;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let flags: Vec<&str> = args.iter().filter(|a| a.starts_with("--")).map(String::as_str).collect();
    let pos: Vec<&str> = args.iter().skip(1).filter(|a| !a.starts_with("--")).map(String::as_str).collect();
    if pos.len() < 3 {
        eprintln!("gen <fonts dir> <out dir> <photos> [seed] [--val] [--show]");
        process::exit(2);
    }
    let (fonts, out_dir) = (PathBuf::from(pos[0]), PathBuf::from(pos[1]));
    let photos: u32 = pos[2].parse().expect("photos is a number");
    let seed: u32 = pos.get(3).map_or(1, |s| s.parse().expect("seed is a number"));
    let val = flags.contains(&"--val");

    let chosen: Vec<&str> = FAMILIES.iter().copied().filter(|f| HELD_OUT.contains(f) == val).collect();
    let families: Vec<Vec<Face>> = load_families(&fonts, &chosen).into_iter().filter(|f| !f.is_empty()).collect();
    eprintln!("{} families, {} faces", families.len(), families.iter().map(Vec::len).sum::<usize>());
    fs::create_dir_all(&out_dir).expect("create out dir");
    let show = flags.contains(&"--show").then(|| out_dir.clone());

    let workers = thread::available_parallelism().map_or(4, |n| n.get()) as u32;
    let results: Vec<(Vec<u8>, Tally)> = thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|k| {
                let families = &families;
                let show = show.as_deref();
                s.spawn(move || {
                    let mut solver = Solver::new();
                    let (mut out, mut tally) = (Vec::new(), Tally::default());
                    for n in (k..photos).step_by(workers as usize) {
                        let show = show.filter(|_| n < 4);
                        one_photo(seed.wrapping_add(n), families, &mut solver, show, &mut out, &mut tally);
                    }
                    (out, tally)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker")).collect()
    });

    let mut all = Vec::new();
    let mut total = Tally::default();
    for (out, t) in results {
        all.extend(out);
        total.photos += t.photos;
        total.missed += t.missed;
        for c in 0..10 {
            total.per_class[c] += t.per_class[c];
        }
    }
    let file = out_dir.join(if val { "val.bin" } else { "train.bin" });
    fs::write(&file, &all).expect("write data");
    eprintln!(
        "{} photos, grid missed in {}, {} cells per class 0-9: {:?} -> {}",
        total.photos,
        total.missed,
        all.len() / data::RECORD,
        total.per_class,
        file.display()
    );
}
