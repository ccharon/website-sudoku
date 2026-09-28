use sudoku_solver::Random;

pub struct Rng(Random);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(Random::from_seed(seed))
    }

    /// A number in [0, 1).
    pub fn f(&mut self) -> f64 {
        self.0.next_f64()
    }

    pub fn range(&mut self, from: f64, to: f64) -> f64 {
        from + (to - from) * self.f()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.f() * n as f64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f() < p
    }

    /// Standard normal, Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.f();
        let v = self.f();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}
