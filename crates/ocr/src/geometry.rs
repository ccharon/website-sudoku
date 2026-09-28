use crate::js;

pub type Point = [f64; 2];

/// Corners top left, top right, bottom right, bottom left.
pub type Quad = [Point; 4];

const BEND_START: f64 = 0.5; // pixels a hull point must stand out to be kept
const BEND_LIMIT: f64 = 1e4; // past this the hull does not thin out any further
const SIDE_ENDS: f64 = 0.08; // a point this near either end belongs to no single side
const MIN_SIDE: f64 = 8.0; // pixels, a shorter side is not a photographed grid
const SIDE_RATIO: f64 = 2.5; // longest side over shortest, above this it is not one

pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[sorted.len() >> 1]
}

fn triangle_area(a: Point, b: Point, c: Point) -> f64 {
    ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
}

fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull of a point set, counter clockwise, monotone chain.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    if p.len() < 3 {
        return p;
    }
    let half = |list: &mut dyn Iterator<Item = &Point>| {
        let mut out: Vec<Point> = Vec::new();
        for &q in list {
            while out.len() >= 2 && cross(out[out.len() - 2], out[out.len() - 1], q) <= 0.0 {
                out.pop();
            }
            out.push(q);
        }
        out.pop();
        out
    };
    let mut hull = half(&mut p.iter());
    hull.extend(half(&mut p.iter().rev()));
    hull
}

/// Drops hull points that barely bend, down to at most limit of them.
pub fn simplify_hull(hull: Vec<Point>, limit: usize) -> Vec<Point> {
    let mut out = hull;
    let mut tolerance = BEND_START;
    while out.len() > limit {
        let n = out.len();
        let mut kept = Vec::new();
        for i in 0..n {
            let a = out[(i + n - 1) % n];
            let b = out[i];
            let c = out[(i + 1) % n];
            let span = js::hypot(c[0] - a[0], c[1] - a[1]);
            let span = if span == 0.0 || span.is_nan() { 1.0 } else { span };
            let away = ((c[0] - a[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (c[1] - a[1])).abs() / span;
            if away >= tolerance {
                kept.push(b);
            }
        }
        if kept.len() < 4 || kept.len() == n {
            tolerance *= 2.0;
            if tolerance > BEND_LIMIT {
                break;
            }
            continue;
        }
        out = kept;
    }
    out
}

/// The four hull points spanning the largest area. Unlike the extremes along
/// the diagonals this survives rotation and a stray spur on the outline.
pub fn max_area_quad(hull: &[Point]) -> Option<Quad> {
    let n = hull.len();
    if n < 4 {
        return None;
    }
    if n == 4 {
        return Some([hull[0], hull[1], hull[2], hull[3]]);
    }
    let mut best = None;
    let mut best_area = -1.0;
    for i in 0..n {
        for j in i + 2..n {
            let (mut left, mut left_at) = (-1.0, None);
            let (mut right, mut right_at) = (-1.0, None);
            for k in i + 1..j {
                let t = triangle_area(hull[i], hull[k], hull[j]);
                if t > left {
                    left = t;
                    left_at = Some(k);
                }
            }
            for k in j + 1..n + i {
                let t = triangle_area(hull[j], hull[k % n], hull[i]);
                if t > right {
                    right = t;
                    right_at = Some(k % n);
                }
            }
            let (Some(l), Some(r)) = (left_at, right_at) else { continue };
            if left + right > best_area {
                best_area = left + right;
                best = Some([hull[i], hull[l], hull[j], hull[r]]);
            }
        }
    }
    best
}

pub fn distance_to_side(p: Point, a: Point, b: Point) -> f64 {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let len = vx * vx + vy * vy;
    let t = if len == 0.0 { 0.0 } else { ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len };
    if t < SIDE_ENDS || t > 1.0 - SIDE_ENDS {
        return f64::INFINITY; // corners belong to two sides
    }
    ((p[0] - a[0]) * vy - (p[1] - a[1]) * vx).abs() / len.sqrt()
}

/// Line of least squared distance through the points, as nx*x + ny*y = c.
pub fn fit_line(points: &[Point]) -> [f64; 3] {
    let n = points.len() as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for p in points {
        sx += p[0];
        sy += p[1];
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for p in points {
        let dx = p[0] - mx;
        let dy = p[1] - my;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let nx = -angle.sin();
    let ny = angle.cos();
    [nx, ny, nx * mx + ny * my]
}

pub fn cross_lines(a: [f64; 3], b: [f64; 3]) -> Option<Point> {
    let det = a[0] * b[1] - b[0] * a[1];
    if det.abs() < 1e-9 {
        return None;
    }
    Some([(a[2] * b[1] - b[2] * a[1]) / det, (a[0] * b[2] - b[0] * a[2]) / det])
}

/// Brings the four corners into the order top left, top right, bottom right, bottom left.
pub fn order_quad(quad: Quad) -> Quad {
    let mut area = 0.0;
    for i in 0..4 {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        area += a[0] * b[1] - b[0] * a[1];
    }
    let mut ring = quad;
    if area < 0.0 {
        ring.reverse();
    }
    let mut first = 0;
    for i in 1..4 {
        if ring[i][0] + ring[i][1] < ring[first][0] + ring[first][1] {
            first = i;
        }
    }
    [ring[first], ring[(first + 1) % 4], ring[(first + 2) % 4], ring[(first + 3) % 4]]
}

/// The four side lengths of a quadrilateral, from corner i to corner i + 1.
pub fn side_lengths(quad: &Quad) -> [f64; 4] {
    std::array::from_fn(|i| {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        js::hypot(b[0] - a[0], b[1] - a[1])
    })
}

/// Rejects quadrilaterals too lopsided to be a photographed grid.
pub fn plausible_quad(quad: &Quad) -> bool {
    let sides = side_lengths(quad);
    let shortest = js::min(&sides);
    shortest > MIN_SIDE && js::max(&sides) / shortest < SIDE_RATIO
}

/// Mean side length of the quadrilateral, in pixels of the photo.
pub fn side_of(quad: &Quad) -> f64 {
    side_lengths(quad).iter().sum::<f64>() / 4.0
}

/// Maps the unit square onto a quadrilateral.
struct Perspective {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
}

impl Perspective {
    /// After Heckbert.
    fn new(quad: &Quad) -> Self {
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *quad;
        let (dx1, dx2, dx3) = (x1 - x2, x3 - x2, x0 - x1 + x2 - x3);
        let (dy1, dy2, dy3) = (y1 - y2, y3 - y2, y0 - y1 + y2 - y3);

        if dx3 == 0.0 && dy3 == 0.0 {
            return Perspective { a: x1 - x0, b: x2 - x1, c: x0, d: y1 - y0, e: y2 - y1, f: y0, g: 0.0, h: 0.0 };
        }
        let det = dx1 * dy2 - dy1 * dx2;
        let g = (dx3 * dy2 - dy3 * dx2) / det;
        let h = (dx1 * dy3 - dy1 * dx3) / det;
        Perspective {
            a: x1 - x0 + g * x1,
            b: x3 - x0 + h * x3,
            c: x0,
            d: y1 - y0 + g * y1,
            e: y3 - y0 + h * y3,
            f: y0,
            g,
            h,
        }
    }

    fn map(&self, u: f64, v: f64) -> Point {
        let den = self.g * u + self.h * v + 1.0;
        [(self.a * u + self.b * v + self.c) / den, (self.d * u + self.e * v + self.f) / den]
    }
}

fn sample(gray: &[u8], w: usize, h: usize, x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let a = gray[y0 * w + x0] as f64;
    let b = gray[y0 * w + x1] as f64;
    let c = gray[y1 * w + x0] as f64;
    let d = gray[y1 * w + x1] as f64;
    a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
}

/// Samples the photo along the straightened grid, bilinear.
pub fn rectify(gray: &[u8], w: usize, h: usize, quad: &Quad, size: usize) -> Vec<u8> {
    let m = Perspective::new(quad);
    let mut out = vec![0; size * size];
    for j in 0..size {
        let v = (j as f64 + 0.5) / size as f64;
        for i in 0..size {
            let u = (i as f64 + 0.5) / size as f64;
            let [x, y] = m.map(u, v);
            out[j * size + i] = js::clamp_u8(sample(gray, w, h, x, y));
        }
    }
    out
}

/// An axis aligned rectangle in pixels of a straightened image.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// The corners of the rectangle, in the order rectify expects.
pub fn rect_quad(r: &Rect) -> Quad {
    [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]
}

/// Carries points of the straightened image back into the photo.
pub fn to_photo(quad: &Quad, points: &Quad, size: usize) -> Quad {
    let m = Perspective::new(quad);
    points.map(|p| m.map(p[0] / size as f64, p[1] / size as f64))
}
