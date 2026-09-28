//! Trains the digit network on the output of gen and writes the weights
//! with the best accuracy on the validation set.
//!
//!   train <data dir> <weights out> [epochs]

use std::path::PathBuf;
use std::time::Instant;
use std::{env, fs, process, thread};

use sudoku_ocr::net::{LAYOUT, Net};
use sudoku_train::data::{self, Sample};
use sudoku_train::grad::{Adam, backward, init};
use sudoku_train::rng::Rng;

const BATCH: usize = 128;
const LEARNING_RATE: f32 = 2e-3;
const DECAY: f32 = 0.8; // learning rate factor per epoch

/// Share of correct answers, and the confusions as (truth, answer, count).
fn evaluate(net: &Net, samples: &[Sample]) -> (f64, Vec<(usize, usize, usize)>) {
    let mut confusion = [[0usize; 10]; 10];
    for s in samples {
        let p = net.classify(&s.mask, &s.shape);
        let answer = (0..10).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0);
        confusion[s.label][answer] += 1;
    }
    let right: usize = (0..10).map(|k| confusion[k][k]).sum();
    let mut wrong: Vec<_> = (0..10)
        .flat_map(|t| (0..10).map(move |a| (t, a)))
        .filter(|&(t, a)| t != a)
        .map(|(t, a)| (t, a, confusion[t][a]))
        .filter(|c| c.2 > 0)
        .collect();
    wrong.sort_by_key(|c| std::cmp::Reverse(c.2));
    (right as f64 / samples.len().max(1) as f64, wrong)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("train <data dir> <weights out> [epochs]");
        process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let epochs: usize = args.get(3).map_or(12, |s| s.parse().expect("epochs is a number"));

    let train = data::load(&dir.join("train.bin")).expect("train.bin");
    let val = data::load(&dir.join("val.bin")).expect("val.bin");
    eprintln!("{} training cells, {} validation cells", train.len(), val.len());

    let mut rng = Rng::new(7);
    let mut net = init(&mut rng);
    let mut adam = Adam::new(LAYOUT.len);
    let workers = thread::available_parallelism().map_or(4, |n| n.get());
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut best = -1.0;
    let mut lr = LEARNING_RATE;

    for epoch in 1..=epochs {
        let started = Instant::now();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut loss_sum = 0.0;
        for batch in order.chunks(BATCH) {
            let parts: Vec<(Vec<f32>, f32)> = thread::scope(|s| {
                let net = &net;
                let train = &train;
                batch
                    .chunks(batch.len().div_ceil(workers))
                    .map(|part| {
                        s.spawn(move || {
                            let mut grad = vec![0.0; LAYOUT.len];
                            let mut loss = 0.0;
                            for &i in part {
                                let sample = &train[i];
                                loss += backward(net, &net.trace(&sample.mask, &sample.shape), sample.label, &mut grad);
                            }
                            (grad, loss)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().expect("worker"))
                    .collect()
            });
            let mut grad = vec![0.0; LAYOUT.len];
            for (g, loss) in parts {
                loss_sum += loss as f64;
                for (a, b) in grad.iter_mut().zip(g) {
                    *a += b / batch.len() as f32;
                }
            }
            adam.update(&mut net.params, &grad, lr);
        }
        lr *= DECAY;

        let (accuracy, wrong) = evaluate(&net, &val);
        let top: Vec<String> = wrong.iter().take(6).map(|(t, a, n)| format!("{t}->{a} {n}")).collect();
        eprintln!(
            "epoch {epoch:2}: loss {:.4}, validation {:.3}%, {:.0}s, most confused {}",
            loss_sum / train.len() as f64,
            accuracy * 100.0,
            started.elapsed().as_secs_f64(),
            top.join(", ")
        );
        if accuracy > best {
            best = accuracy;
            fs::write(&out, net.to_le_bytes()).expect("write weights");
        }
    }
    eprintln!("best validation {:.3}% -> {}", best * 100.0, out.display());
}
