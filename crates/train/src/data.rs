//! A data set is a file of records, each one label byte, the 32×32 mask
//! packed into 128 bytes, eight pixels per byte, first pixel in the highest
//! bit, and the shape measures as f32, little endian.

use std::fs;
use std::io;
use std::path::Path;

use sudoku_ocr::{NORM, SHAPE};

pub const MASK: usize = NORM * NORM;
pub const RECORD: usize = 1 + MASK / 8 + SHAPE * 4;

pub fn push(out: &mut Vec<u8>, label: u8, mask: &[u8], shape: &[f32; SHAPE]) {
    out.push(label);
    for chunk in mask.chunks(8) {
        out.push(chunk.iter().fold(0, |acc, &v| (acc << 1) | (v & 1)));
    }
    for v in shape {
        out.extend(v.to_le_bytes());
    }
}

pub struct Sample {
    pub label: usize,
    pub mask: [u8; MASK],
    pub shape: [f32; SHAPE],
}

pub fn load(path: &Path) -> io::Result<Vec<Sample>> {
    let bytes = fs::read(path)?;
    if bytes.len() % RECORD != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: not a whole number of records", path.display())));
    }
    Ok(bytes
        .chunks(RECORD)
        .map(|r| {
            let mut mask = [0; MASK];
            for (i, m) in mask.iter_mut().enumerate() {
                *m = (r[1 + i / 8] >> (7 - i % 8)) & 1;
            }
            let at = 1 + MASK / 8;
            let shape = std::array::from_fn(|k| f32::from_le_bytes(r[at + 4 * k..at + 4 * k + 4].try_into().unwrap()));
            Sample { label: r[0] as usize, mask, shape }
        })
        .collect())
}
