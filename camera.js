"use strict";

/*
 * Reads a Sudoku from a photo. The sections below follow the way through:
 *
 *   Image and ink     grey plane, adaptive threshold, connected patches
 *   Geometry          hulls, lines, quadrilaterals, the perspective map
 *   Grid lines        where the lines of a straightened grid lie
 *   Grid detection    which ink patch is the Sudoku, and its four corners
 *   Cells             one digit per cell, freed and normalised
 *   Shape matching    distance between two digit shapes
 *   Templates         the digits 1 to 9, rendered from the browser's fonts
 *   Reading           straighten once, then read the digits off it
 *   Repair            the solver checks the reading and mends it
 *   Capture           pixels out of the camera or a picked file
 *   Display           the viewfinder and what it reports
 */

(function () {

const S = globalThis.Sudoku;

/* ------------------------------------------------------------- Image and ink */

const INK_BIAS = 0.9;       // darker than this share of the local mean counts as ink
const INK_FIND = 0.88;      // threshold of the pass that looks for the outline
const THIN_DROP = 0.12;     // the stricter pass, used for counters, sits this far below
const THIN_KEEP = 0.6;      // ink the stricter pass must retain to be believed
const INK_TRIES = [0.9, 0.96, 0.84];    // retried in this order until the reading solves

/** Returns the luma plane of an ImageData. */
function toGray(image) {
	const n = image.width * image.height;
	const gray = new Uint8ClampedArray(n);
	const d = image.data;
	for (let i = 0; i < n; i++) {
		gray[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;
	}
	return gray;
}

/** The grey plane with light and dark exchanged, for a screen in dark mode. */
function invertPlane(gray) {
	const out = new Uint8ClampedArray(gray.length);
	for (let i = 0; i < gray.length; i++) out[i] = 255 - gray[i];
	return out;
}

/** Sums of a grey plane, one row and column larger than the image. */
function integral(gray, w, h) {
	const sum = new Float64Array((w + 1) * (h + 1));
	const stride = w + 1;
	for (let y = 0; y < h; y++) {
		let row = 0;
		for (let x = 0; x < w; x++) {
			row += gray[y * w + x];
			sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
		}
	}
	return sum;
}

/**
 * Marks every pixel darker than the mean of its neighbourhood. A global
 * threshold fails on photos because of shadows and uneven light.
 */
function threshold(gray, w, h, radius, bias) {
	const sum = integral(gray, w, h);
	const mask = new Uint8Array(w * h);
	const stride = w + 1;
	for (let y = 0; y < h; y++) {
		const y0 = Math.max(0, y - radius);
		const y1 = Math.min(h - 1, y + radius);
		for (let x = 0; x < w; x++) {
			const x0 = Math.max(0, x - radius);
			const x1 = Math.min(w - 1, x + radius);
			const count = (x1 - x0 + 1) * (y1 - y0 + 1);
			const area = sum[(y1 + 1) * stride + x1 + 1] - sum[y0 * stride + x1 + 1] -
				sum[(y1 + 1) * stride + x0] + sum[y0 * stride + x0];
			mask[y * w + x] = gray[y * w + x] * count < area * bias ? 1 : 0;
		}
	}
	return mask;
}

/** An empty bounding box, to be grown pixel by pixel. */
function newBox() {
	return { x0: Infinity, y0: Infinity, x1: -1, y1: -1 };
}

function growBox(box, x, y) {
	if (x < box.x0) box.x0 = x;
	if (x > box.x1) box.x1 = x;
	if (y < box.y0) box.y0 = y;
	if (y > box.y1) box.y1 = y;
}

function boxWidth(box) { return box.x1 - box.x0 + 1; }

function boxHeight(box) { return box.y1 - box.y0 + 1; }

/** Copies a rectangle out of a mask into a plane of its own. */
function copyRect(mask, stride, x0, y0, w, h) {
	const out = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) out[y * w + x] = mask[(y0 + y) * stride + x0 + x];
	}
	return out;
}

/**
 * Labels the connected patches of ink, eight-connected. Returns one entry per
 * label with its pixel count and bounding box.
 */
function components(mask, w, h) {
	const labels = new Int32Array(w * h).fill(-1);
	const stack = new Int32Array(w * h);
	const boxes = [];
	for (let start = 0; start < w * h; start++) {
		if (mask[start] === 0 || labels[start] >= 0) continue;
		const box = newBox();
		box.id = boxes.length;
		box.count = 0;
		let top = 0;
		stack[top++] = start;
		labels[start] = box.id;
		while (top > 0) {
			const p = stack[--top];
			const x = p % w;
			const y = (p - x) / w;
			box.count++;
			growBox(box, x, y);
			for (let dy = -1; dy <= 1; dy++) {
				const ny = y + dy;
				if (ny < 0 || ny >= h) continue;
				for (let dx = -1; dx <= 1; dx++) {
					const nx = x + dx;
					if (nx < 0 || nx >= w) continue;
					const q = ny * w + nx;
					if (mask[q] === 1 && labels[q] < 0) {
						labels[q] = box.id;
						stack[top++] = q;
					}
				}
			}
		}
		boxes.push(box);
	}
	return { labels: labels, boxes: boxes };
}

/* ------------------------------------------------------------------ Geometry */

function median(values) {
	const sorted = values.slice().sort(function (a, b) { return a - b; });
	return sorted[sorted.length >> 1];
}

function triangleArea(a, b, c) {
	return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
}

/** Convex hull of a point set, counter clockwise, monotone chain. */
function convexHull(points) {
	const p = points.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
	if (p.length < 3) return p;
	const cross = function (o, a, b) {
		return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
	};
	const half = function (list) {
		const out = [];
		for (const q of list) {
			while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
			out.push(q);
		}
		out.pop();
		return out;
	};
	return half(p).concat(half(p.reverse()));
}

/** Drops hull points that barely bend, down to at most limit of them. */
function simplifyHull(hull, limit) {
	let out = hull;
	let tolerance = 0.5;
	while (out.length > limit) {
		const kept = [];
		for (let i = 0; i < out.length; i++) {
			const a = out[(i + out.length - 1) % out.length];
			const b = out[i];
			const c = out[(i + 1) % out.length];
			const span = Math.hypot(c[0] - a[0], c[1] - a[1]);
			const away = Math.abs((c[0] - a[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (c[1] - a[1])) / (span || 1);
			if (away >= tolerance) kept.push(b);
		}
		if (kept.length < 4 || kept.length === out.length) {
			tolerance *= 2;
			if (tolerance > 1e4) break;
			continue;
		}
		out = kept;
	}
	return out;
}

/**
 * The four hull points spanning the largest area. Unlike the extremes along
 * the diagonals this survives rotation and a stray spur on the outline.
 */
function maxAreaQuad(hull) {
	const n = hull.length;
	if (n < 4) return null;
	if (n === 4) return hull.slice();
	let best = null;
	let bestArea = -1;
	for (let i = 0; i < n; i++) {
		for (let j = i + 2; j < n; j++) {
			let left = -1, leftAt = -1, right = -1, rightAt = -1;
			for (let k = i + 1; k < j; k++) {
				const t = triangleArea(hull[i], hull[k], hull[j]);
				if (t > left) { left = t; leftAt = k; }
			}
			for (let k = j + 1; k < n + i; k++) {
				const t = triangleArea(hull[j], hull[k % n], hull[i]);
				if (t > right) { right = t; rightAt = k % n; }
			}
			if (leftAt < 0 || rightAt < 0) continue;
			if (left + right > bestArea) {
				bestArea = left + right;
				best = [hull[i], hull[leftAt], hull[j], hull[rightAt]];
			}
		}
	}
	return best;
}

function distanceToSide(p, a, b) {
	const vx = b[0] - a[0];
	const vy = b[1] - a[1];
	const len = vx * vx + vy * vy;
	const t = len === 0 ? 0 : ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len;
	if (t < 0.08 || t > 0.92) return Infinity;    // corners belong to two sides
	return Math.abs((p[0] - a[0]) * vy - (p[1] - a[1]) * vx) / Math.sqrt(len);
}

/** Line of least squared distance through the points, as nx*x + ny*y = c. */
function fitLine(points) {
	let sx = 0;
	let sy = 0;
	for (const p of points) { sx += p[0]; sy += p[1]; }
	const mx = sx / points.length;
	const my = sy / points.length;
	let xx = 0, xy = 0, yy = 0;
	for (const p of points) {
		const dx = p[0] - mx;
		const dy = p[1] - my;
		xx += dx * dx; xy += dx * dy; yy += dy * dy;
	}
	const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
	const nx = -Math.sin(angle);
	const ny = Math.cos(angle);
	return [nx, ny, nx * mx + ny * my];
}

function crossLines(a, b) {
	const det = a[0] * b[1] - b[0] * a[1];
	if (Math.abs(det) < 1e-9) return null;
	return [(a[2] * b[1] - b[2] * a[1]) / det, (a[0] * b[2] - b[0] * a[2]) / det];
}

/** Brings the four corners into the order top left, top right, bottom right, bottom left. */
function orderQuad(quad) {
	let area = 0;
	for (let i = 0; i < 4; i++) {
		const a = quad[i];
		const b = quad[(i + 1) % 4];
		area += a[0] * b[1] - b[0] * a[1];
	}
	const ring = area < 0 ? quad.slice().reverse() : quad.slice();
	let first = 0;
	for (let i = 1; i < 4; i++) {
		if (ring[i][0] + ring[i][1] < ring[first][0] + ring[first][1]) first = i;
	}
	return [ring[first], ring[(first + 1) % 4], ring[(first + 2) % 4], ring[(first + 3) % 4]];
}

/** The four side lengths of a quadrilateral, from corner i to corner i + 1. */
function sideLengths(quad) {
	const out = [];
	for (let i = 0; i < 4; i++) {
		const a = quad[i];
		const b = quad[(i + 1) % 4];
		out.push(Math.hypot(b[0] - a[0], b[1] - a[1]));
	}
	return out;
}

/** Rejects quadrilaterals too lopsided to be a photographed grid. */
function plausibleQuad(quad) {
	const sides = sideLengths(quad);
	const shortest = Math.min(...sides);
	return shortest > 8 && Math.max(...sides) / shortest < 2.5;
}

/** Mean side length of the quadrilateral, in pixels of the photo. */
function sideOf(quad) {
	let sum = 0;
	for (const len of sideLengths(quad)) sum += len;
	return sum / 4;
}

/** Maps the unit square onto the quadrilateral, after Heckbert. */
function unitToQuad(quad) {
	const x0 = quad[0][0], y0 = quad[0][1];
	const x1 = quad[1][0], y1 = quad[1][1];
	const x2 = quad[2][0], y2 = quad[2][1];
	const x3 = quad[3][0], y3 = quad[3][1];
	const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
	const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;

	if (dx3 === 0 && dy3 === 0) {
		return { a: x1 - x0, b: x2 - x1, c: x0, d: y1 - y0, e: y2 - y1, f: y0, g: 0, h: 0 };
	}
	const det = dx1 * dy2 - dy1 * dx2;
	const g = (dx3 * dy2 - dy3 * dx2) / det;
	const h = (dx1 * dy3 - dy1 * dx3) / det;
	return {
		a: x1 - x0 + g * x1, b: x3 - x0 + h * x3, c: x0,
		d: y1 - y0 + g * y1, e: y3 - y0 + h * y3, f: y0,
		g: g, h: h
	};
}

function sample(gray, w, h, x, y) {
	if (x < 0) x = 0; else if (x > w - 1) x = w - 1;
	if (y < 0) y = 0; else if (y > h - 1) y = h - 1;
	const x0 = Math.floor(x), y0 = Math.floor(y);
	const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
	const fx = x - x0, fy = y - y0;
	const a = gray[y0 * w + x0], b = gray[y0 * w + x1];
	const c = gray[y1 * w + x0], d = gray[y1 * w + x1];
	return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

/** Samples the photo along the straightened grid, bilinear. */
function rectify(gray, w, h, quad, size) {
	const m = unitToQuad(quad);
	const out = new Uint8ClampedArray(size * size);
	for (let j = 0; j < size; j++) {
		const v = (j + 0.5) / size;
		for (let i = 0; i < size; i++) {
			const u = (i + 0.5) / size;
			const den = m.g * u + m.h * v + 1;
			out[j * size + i] = sample(gray, w, h,
				(m.a * u + m.b * v + m.c) / den,
				(m.d * u + m.e * v + m.f) / den);
		}
	}
	return out;
}

/** The corners of an axis aligned rectangle, in the order rectify expects. */
function rectQuad(rect) {
	return [[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x1, rect.y1], [rect.x0, rect.y1]];
}

/* ---------------------------------------------------------------- Grid lines */

/**
 * Index steps of one direction: from one band to the next, and from one pixel
 * of a line to the next along it. With along set the bands are columns.
 */
function lineSteps(size, along) {
	return along ? { band: 1, walk: size } : { band: size, walk: 1 };
}

/** Centres of the runs of pixels that carry a line across the whole image. */
function lineBands(ink, size, along) {
	const { band, walk } = lineSteps(size, along);
	const bands = [];
	let start = -1;
	for (let i = 0; i < size; i++) {
		let count = 0;
		for (let k = 0; k < size; k++) {
			count += ink[i * band + k * walk];
		}
		const isLine = count >= size * 0.6;
		if (isLine && start < 0) start = i;
		if (!isLine && start >= 0) {
			if (i - start <= size * 0.04) bands.push((start + i - 1) / 2);
			start = -1;
		}
	}
	if (start >= 0 && size - start <= size * 0.04) bands.push((start + size - 1) / 2);
	return bands;
}

/**
 * Length and position of the longest unbroken stretch of the line through this
 * band. Plain first and last ink would catch every crossing line instead.
 */
function bandExtent(ink, size, centre, along) {
	const { band, walk } = lineSteps(size, along);
	const from = Math.max(0, Math.round(centre) - 1);
	const to = Math.min(size - 1, Math.round(centre) + 1);
	let bestFirst = -1;
	let bestLast = -1;
	let best = -1;
	let first = -1;
	let last = -1;
	let gap = 0;

	function close() {
		if (first >= 0 && last - first > best) {
			best = last - first;
			bestFirst = first;
			bestLast = last;
		}
		first = -1;
	}

	for (let k = 0; k < size; k++) {
		let hit = 0;
		for (let i = from; i <= to; i++) {
			hit |= ink[i * band + k * walk];
		}
		if (hit) {
			if (first < 0) first = k;
			last = k;
			gap = 0;
		} else if (first >= 0 && ++gap > 3) {
			close();
		}
	}
	close();
	return [bestFirst, bestLast];
}

/** Where the inner lines of one direction begin and end, across it. */
function innerExtent(ink, size, bands, along) {
	const firsts = [];
	const lasts = [];
	for (const centre of bands.slice(1, -1)) {
		const span = bandExtent(ink, size, centre, along);
		if (span[0] < 0) continue;
		firsts.push(span[0]);
		lasts.push(span[1]);
	}
	if (firsts.length === 0) return null;
	return [median(firsts), median(lasts) + 1];
}

/**
 * Rates how much the area inside the quadrilateral looks like a Sudoku: ten
 * evenly spaced lines in both directions, spanning the whole of it. Without
 * this the largest patch wins, which on a tablet is the bezel of the screen.
 */
function bandScore(ink, size) {
	let score = 1;
	for (const along of [true, false]) {
		const bands = lineBands(ink, size, along);
		if (bands.length < 8) return 0;
		const steps = [];
		for (let i = 1; i < bands.length; i++) steps.push(bands[i] - bands[i - 1]);
		const step = median(steps);
		if (!(step > 0)) return 0;
		const reach = bands[bands.length - 1] - bands[0];
		const cells = Math.round(reach / step);
		if (cells < 7 || cells > 11) return 0;
		let drift = 0;
		for (const d of steps) drift += Math.abs(d - step);
		drift /= steps.length * step;
		score *= Math.max(0, 1 - 4 * drift) * (reach / size) * (cells === S.SIZE ? 1 : 0.6);
	}
	return score;
}

/**
 * Narrows the straightened image down to the 9x9 grid. Printed puzzles often
 * carry a title box that shares its bottom edge with the grid, so the outline
 * encloses ten rows. The inner lines betray the real extent, because they stop
 * at the grid. Whether the cut is an improvement is decided by the caller.
 */
function gridBounds(ink, size) {
	const box = { x0: 0, y0: 0, x1: size, y1: size };
	for (const along of [true, false]) {
		const bands = lineBands(ink, size, along);
		if (bands.length < 4) return null;
		const span = innerExtent(ink, size, bands, along);
		if (span === null) return null;
		if (along) { box.y0 = span[0]; box.y1 = span[1]; }
		else { box.x0 = span[0]; box.x1 = span[1]; }
	}
	if (box.x1 - box.x0 < size * 0.5 || box.y1 - box.y0 < size * 0.5) return null;
	const trimmed = box.x0 > size * 0.02 || box.y0 > size * 0.02 ||
		box.x1 < size * 0.98 || box.y1 < size * 0.98;
	return trimmed ? box : null;
}

/* ------------------------------------------------------------ Grid detection */

const CANDIDATES = 6;       // ink patches examined before settling on one
const SCORE_CELL = 32;      // cell size of the cheap pass that rates a candidate
const WEAK_GRID = 0.02;     // below this the patch is tried again without its attachments
const RUN_SHARE = 0.3;      // share of the patch a run must span to pass as a line

/**
 * Fits a line to each of the four sides and intersects them. A thin or blurred
 * outline loses its corners, but the long sides survive, so the corner follows
 * from them instead of from a pixel that may not be there.
 */
function refineQuad(quad, points) {
	let current = quad;
	for (let pass = 0; pass < 2; pass++) {
		const near = sideOf(current) * 0.06;
		const sides = [[], [], [], []];
		for (const p of points) {
			let at = -1;
			let best = near;
			for (let i = 0; i < 4; i++) {
				const d = distanceToSide(p, current[i], current[(i + 1) % 4]);
				if (d < best) { best = d; at = i; }
			}
			if (at >= 0) sides[at].push(p);
		}
		const lines = [];
		for (const side of sides) {
			if (side.length < 12) return current;
			lines.push(fitLine(side));
		}
		const next = [];
		for (let i = 0; i < 4; i++) {
			const corner = crossLines(lines[(i + 3) % 4], lines[i]);
			if (corner === null) return current;
			next.push(corner);
		}
		current = next;
	}
	return current;
}

/** The outermost pixel of each row and column, the only ones that lie on a side. */
function outlinePoints(labels, w, box) {
	const points = [];
	for (let y = box.y0; y <= box.y1; y++) {
		let from = -1;
		let to = -1;
		for (let x = box.x0; x <= box.x1; x++) {
			if (labels[y * w + x] !== box.id) continue;
			if (from < 0) from = x;
			to = x;
		}
		if (from >= 0) {
			points.push([from, y]);
			if (to !== from) points.push([to, y]);
		}
	}
	for (let x = box.x0; x <= box.x1; x++) {
		let from = -1;
		let to = -1;
		for (let y = box.y0; y <= box.y1; y++) {
			if (labels[y * w + x] !== box.id) continue;
			if (from < 0) from = y;
			to = y;
		}
		if (from >= 0) {
			points.push([x, from]);
			if (to !== from) points.push([x, to]);
		}
	}
	return points;
}

/** Hull, largest quadrilateral, then the sides fitted to the points again. */
function quadFromPoints(points) {
	if (points.length < 8) return null;
	const corners = maxAreaQuad(simplifyHull(convexHull(points), 48));
	if (corners === null) return null;
	const quad = orderQuad(refineQuad(orderQuad(corners), points));
	return plausibleQuad(quad) ? quad : null;
}

/** The four corners of one ink patch, or null if it has too few pixels. */
function quadOf(labels, w, box) {
	return quadFromPoints(outlinePoints(labels, w, box));
}

/**
 * The outline of the patch with only the pixels that lie in a long run, so
 * nothing but its lines is left. A drawing that touches the frame drops out,
 * and with it the corner it would pull away.
 */
function latticePoints(labels, w, box) {
	const bw = boxWidth(box);
	const bh = boxHeight(box);
	const need = Math.round(Math.max(bw, bh) * RUN_SHARE);
	const keep = new Uint8Array(bw * bh);

	for (let y = 0; y < bh; y++) {
		let run = 0;
		for (let x = 0; x <= bw; x++) {
			if (x < bw && labels[(box.y0 + y) * w + box.x0 + x] === box.id) { run++; continue; }
			if (run >= need) for (let k = x - run; k < x; k++) keep[y * bw + k] = 1;
			run = 0;
		}
	}
	for (let x = 0; x < bw; x++) {
		let run = 0;
		for (let y = 0; y <= bh; y++) {
			if (y < bh && labels[(box.y0 + y) * w + box.x0 + x] === box.id) { run++; continue; }
			if (run >= need) for (let k = y - run; k < y; k++) keep[k * bw + x] = 1;
			run = 0;
		}
	}

	const points = outlinePoints(keep, bw, { id: 1, x0: 0, y0: 0, x1: bw - 1, y1: bh - 1 });
	for (const p of points) { p[0] += box.x0; p[1] += box.y0; }
	return points;
}

function gridLikeness(gray, w, h, quad) {
	const probe = SCORE_CELL * 9;
	return bandScore(threshold(rectify(gray, w, h, quad, probe), probe, probe,
		Math.round(SCORE_CELL / 2), INK_BIAS), probe);
}

/**
 * Picks the ink patch that reads as a Sudoku. Candidates are wide, roughly
 * square outlines rather than solid areas; the largest goes first, but the
 * line structure decides.
 */
function findGrid(gray, ink, w, h) {
	const found = components(ink, w, h);
	const minArea = w * h * 0.02;
	const candidates = [];
	for (const box of found.boxes) {
		const bw = boxWidth(box);
		const bh = boxHeight(box);
		const area = bw * bh;
		if (area < minArea) continue;
		if (bw / bh < 0.5 || bw / bh > 2) continue;
		const fill = box.count / area;
		if (fill < 0.02 || fill > 0.6) continue;
		candidates.push({ box: box, area: area });
	}
	candidates.sort(function (a, b) { return b.area - a.area; });

	let best = null;
	for (const candidate of candidates.slice(0, CANDIDATES)) {
		const quad = quadOf(found.labels, w, candidate.box);
		if (quad === null) continue;
		const score = gridLikeness(gray, w, h, quad);
		if (best === null || score > best.score) {
			best = { quad: quad, score: score, box: candidate.box };
		}
		if (score > 0.8) break;
	}
	if (best === null) return null;

	// So little line structure means something is stuck to the grid.
	if (best.score < WEAK_GRID) {
		const quad = quadFromPoints(latticePoints(found.labels, w, best.box));
		if (quad !== null && gridLikeness(gray, w, h, quad) > best.score) return quad;
	}
	return best.quad;
}

/* --------------------------------------------------------------------- Cells */

const CELL_INSET = 0.14;    // share of a cell dropped on each side
const CELL_OFF = 0.36;      // how far off centre a digit may sit inside a cell
const NORM = 32;            // side of a normalised digit bitmap
const NORM_FIT = 24;        // the digit is scaled to fit this box

/**
 * Scales the digit to a fixed box and centres it by its centre of mass, so
 * font size and position inside the cell stop mattering.
 */
function normalise(bitmap, w, box) {
	const bw = boxWidth(box);
	const bh = boxHeight(box);
	const scale = NORM_FIT / Math.max(bw, bh);
	const tw = Math.max(1, Math.min(NORM, Math.round(bw * scale)));
	const th = Math.max(1, Math.min(NORM, Math.round(bh * scale)));

	const small = new Uint8Array(tw * th);
	let cx = 0, cy = 0, count = 0;
	for (let y = 0; y < th; y++) {
		const sy0 = Math.floor(y * bh / th);
		const sy1 = Math.max(Math.floor((y + 1) * bh / th), sy0 + 1);
		for (let x = 0; x < tw; x++) {
			const sx0 = Math.floor(x * bw / tw);
			const sx1 = Math.max(Math.floor((x + 1) * bw / tw), sx0 + 1);
			let on = 0, total = 0;
			for (let yy = sy0; yy < sy1; yy++) {
				for (let xx = sx0; xx < sx1; xx++) {
					on += bitmap[(box.y0 + yy) * w + box.x0 + xx];
					total++;
				}
			}
			if (on / total >= 0.4) {
				small[y * tw + x] = 1;
				cx += x; cy += y; count++;
			}
		}
	}
	if (count === 0) return null;

	const out = new Uint8Array(NORM * NORM);
	const ox = Math.min(Math.max(Math.round(NORM / 2 - cx / count), 0), NORM - tw);
	const oy = Math.min(Math.max(Math.round(NORM / 2 - cy / count), 0), NORM - th);
	for (let y = 0; y < th; y++) {
		for (let x = 0; x < tw; x++) {
			if (small[y * tw + x]) out[(y + oy) * NORM + x + ox] = 1;
		}
	}
	return out;
}

/** Returns the normalised digit bitmap of one cell, or null if it is empty. */
function isolateDigit(ink, thin, size, rect, sourceScale) {
	const w = rect.x1 - rect.x0;
	const h = rect.y1 - rect.y0;
	const sub = copyRect(ink, size, rect.x0, rect.y0, w, h);
	let marked = 0;
	for (let i = 0; i < sub.length; i++) marked += sub[i];
	if (marked < w * h * 0.006) return null;

	const found = components(sub, w, h);
	let best = null;
	for (const box of found.boxes) {
		const bw = boxWidth(box);
		const bh = boxHeight(box);
		if (box.count < 5) continue;
		if (bh < h * 0.22) continue;     // flat leftovers are grid lines
		if (bw > w * 0.92) continue;     // a run across the cell is a line
		// A digit sits in the middle, a piece of a dotted rule sits at the edge.
		if (Math.abs((box.x0 + box.x1) / 2 - w / 2) > w * CELL_OFF) continue;
		if (Math.abs((box.y0 + box.y1) / 2 - h / 2) > h * CELL_OFF) continue;
		if (best === null || box.count > best.count) best = box;
	}
	if (best === null) return null;

	// Too little of it to call a digit. Kept so the solver can still ask.
	const faint = marked < w * h * 0.02 || best.count < 10 || boxHeight(best) < h * 0.3;

	const digit = new Uint8Array(w * h);
	for (let p = 0; p < w * h; p++) digit[p] = found.labels[p] === best.id ? 1 : 0;
	const norm = normalise(digit, w, best);
	if (norm === null) return null;

	/*
	 * Thick ink closes a counter, which only the stricter pass still sees. Weak
	 * ink breaks a loop open, which only the fuller pass still closes. Which of
	 * the two to believe shows in how much ink the stricter pass keeps.
	 */
	const slim = new Uint8Array(w * h);
	let full = 0;
	let kept = 0;
	for (let y = best.y0; y <= best.y1; y++) {
		for (let x = best.x0; x <= best.x1; x++) {
			slim[y * w + x] = thin[(rect.y0 + y) * size + rect.x0 + x];
			full += digit[y * w + x];
			kept += slim[y * w + x];
		}
	}
	const counted = kept > full * THIN_KEEP ? slim : digit;
	const tall = boxHeight(best) * sourceScale;
	return {
		mask: norm,
		holes: holeCount(counted, w, h, minHole(best)),
		trust: Math.min(1, tall / HOLE_SURE),
		faint: faint
	};
}

/**
 * Cuts the straightened grid into cells. The outer share of every cell is
 * dropped so the grid lines fall away.
 */
function cutCells(ink, thin, size, sourceScale) {
	const cell = size / 9;
	const inset = Math.round(cell * CELL_INSET);
	const out = [];
	for (let r = 0; r < 9; r++) {
		for (let c = 0; c < 9; c++) {
			out.push(isolateDigit(ink, thin, size, {
				x0: Math.round(c * cell) + inset,
				y0: Math.round(r * cell) + inset,
				x1: Math.round((c + 1) * cell) - inset,
				y1: Math.round((r + 1) * cell) - inset
			}, sourceScale));
		}
	}
	return out;
}

/* ------------------------------------------------------------ Shape matching */

const HOLE_COST = 0.2;      // penalty per enclosed area two shapes differ by
const HOLE_SURE = 32;       // digit height in source pixels for a fully trusted count

/**
 * Carries the distance of the already visited row and neighbour into every
 * pixel, running from one corner of the bitmap to the opposite one.
 */
function sweep(dist, dir) {
	const first = dir > 0 ? 0 : NORM - 1;
	const past = dir > 0 ? NORM : -1;
	for (let y = first; y !== past; y += dir) {
		const row = y - dir;
		for (let x = first; x !== past; x += dir) {
			const i = y * NORM + x;
			let d = dist[i];
			if (row >= 0 && row < NORM) {
				const j = row * NORM + x;
				if (dist[j] + 3 < d) d = dist[j] + 3;
				if (x > 0 && dist[j - 1] + 4 < d) d = dist[j - 1] + 4;
				if (x < NORM - 1 && dist[j + 1] + 4 < d) d = dist[j + 1] + 4;
			}
			const side = x - dir;
			if (side >= 0 && side < NORM && dist[i - dir] + 3 < d) d = dist[i - dir] + 3;
			dist[i] = d;
		}
	}
}

/** Chamfer distance to the nearest ink pixel, two passes over the bitmap. */
function distanceMap(bitmap) {
	const dist = new Float32Array(NORM * NORM);
	for (let i = 0; i < dist.length; i++) dist[i] = bitmap[i] ? 0 : 1e4;
	sweep(dist, 1);
	sweep(dist, -1);
	for (let i = 0; i < dist.length; i++) dist[i] /= 3;
	return dist;
}

/** A counter smaller than this share of the digit is threshold noise. */
function minHole(box) {
	return Math.max(2, Math.round(boxWidth(box) * boxHeight(box) * 0.004));
}

/**
 * Counts the background patches fully enclosed by ink. Distance alone cannot
 * tell a closed loop from an open hook, which is what separates 5 from 6 and
 * 3 from 8. Counted before the bitmap is scaled down, because scaling closes
 * the counters of bold faces.
 */
function holeCount(bitmap, w, h, limit) {
	const n = w * h;
	const seen = new Uint8Array(n);
	const stack = new Int32Array(n);
	let top = 0;

	function step(q) {
		if (bitmap[q] === 0 && seen[q] === 0) {
			seen[q] = 1;
			stack[top++] = q;
		}
	}

	function flood() {
		let size = 0;
		while (top > 0) {
			const p = stack[--top];
			const x = p % w;
			size++;
			if (x > 0) step(p - 1);
			if (x < w - 1) step(p + 1);
			if (p >= w) step(p - w);
			if (p < n - w) step(p + w);
		}
		return size;
	}

	for (let x = 0; x < w; x++) { step(x); step((h - 1) * w + x); }
	for (let y = 0; y < h; y++) { step(y * w); step(y * w + w - 1); }
	flood();

	let holes = 0;
	for (let start = 0; start < n; start++) {
		if (bitmap[start] === 1 || seen[start] === 1) continue;
		seen[start] = 1;
		stack[top++] = start;
		if (flood() >= limit) holes++;
	}
	return holes;
}

function shapeOf(mask, holes, trust) {
	return { mask: mask, dist: distanceMap(mask), holes: holes, trust: trust };
}

/** Symmetric mean distance between two digit bitmaps. */
function chamfer(a, b) {
	let sum = 0;
	let count = 0;
	for (let i = 0; i < a.mask.length; i++) {
		if (a.mask[i]) { sum += b.dist[i]; count++; }
		if (b.mask[i]) { sum += a.dist[i]; count++; }
	}
	return count === 0 ? Infinity : sum / count;
}

/**
 * Distance plus the difference in enclosed areas, weighted by how well the
 * cell resolves them. A digit twelve pixels tall closes its own counters, so
 * counting them there would decide against the right digit.
 */
function shapeDistance(cell, template) {
	return chamfer(cell, template) +
		HOLE_COST * cell.trust * Math.abs(cell.holes - template.holes);
}

/* ----------------------------------------------------------------- Templates */

const FONTS = [
	"400 96px Arial, Helvetica, sans-serif",
	"700 96px Arial, Helvetica, sans-serif",
	"400 96px Verdana, Tahoma, sans-serif",
	"400 96px Georgia, 'Times New Roman', serif",
	"700 96px Georgia, 'Times New Roman', serif",
	"400 96px 'Courier New', monospace",
	"700 96px 'Courier New', monospace",
	"400 96px 'Arial Narrow', 'Nimbus Sans Narrow', sans-serif",
	"400 96px Tahoma, Geneva, sans-serif",
	"400 96px 'DejaVu Sans', 'Noto Sans', sans-serif",
	"400 96px 'Noto Serif', 'Liberation Serif', serif",
	"400 96px Palatino, 'Palatino Linotype', 'URW Palladio L', serif",
	"400 96px system-ui, sans-serif",
	"700 96px system-ui, sans-serif"
];

let templates = null;

/** A canvas context of the given size, set up for reading the pixels back. */
function drawingContext(width, height) {
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	return canvas.getContext("2d", { willReadFrequently: true });
}

/**
 * Draws the digits 1 to 9 in several fonts and normalises them like the cells.
 * The font rasteriser of the browser takes the place of a trained model.
 */
function buildTemplates() {
	const side = 160;
	const ctx = drawingContext(side, side);
	const out = [];
	const seen = new Set();

	for (const font of FONTS) {
		for (let digit = 1; digit <= 9; digit++) {
			ctx.fillStyle = "#fff";
			ctx.fillRect(0, 0, side, side);
			ctx.fillStyle = "#000";
			ctx.font = font;
			ctx.textAlign = "center";
			ctx.textBaseline = "middle";
			ctx.fillText(String(digit), side / 2, side / 2);

			const data = ctx.getImageData(0, 0, side, side).data;
			const bitmap = new Uint8Array(side * side);
			const box = newBox();
			for (let y = 0; y < side; y++) {
				for (let x = 0; x < side; x++) {
					if (data[(y * side + x) * 4] >= 128) continue;
					bitmap[y * side + x] = 1;
					growBox(box, x, y);
				}
			}
			if (box.x1 < 0) continue;
			const mask = normalise(bitmap, side, box);
			if (mask === null) continue;
			const key = digit + ":" + mask.join("");
			if (seen.has(key)) continue;    // the same face under another name
			seen.add(key);
			const holes = holeCount(bitmap, side, side, minHole(box));
			const shape = shapeOf(mask, holes, 1);
			shape.digit = digit;
			out.push(shape);
		}
	}
	return out;
}

/** Ranks the digits 1 to 9 for one isolated cell. */
function classify(cell) {
	const shape = shapeOf(cell.mask, cell.holes, cell.trust);
	const best = new Float64Array(10).fill(Infinity);
	for (const template of templates) {
		const score = shapeDistance(shape, template);
		if (score < best[template.digit]) best[template.digit] = score;
	}
	const ranked = [];
	for (let d = 1; d <= 9; d++) ranked.push({ digit: d, score: best[d] });
	ranked.sort(function (a, b) { return a.score - b.score; });
	return ranked;
}

/* ------------------------------------------------------------------- Reading */

const WARP_LIVE = 64;       // pixels per cell when reading the camera stream
const WARP_STILL = 80;      // pixels per cell for a single picture

/** The ink mask of a straightened grid at one threshold. */
function inkOf(plane, bias) {
	return threshold(plane.flat, plane.size, plane.size, plane.radius, bias);
}

/** Finds the grid in a grey plane and straightens it, the costly half of a reading. */
function straighten(gray, w, h, cellPx) {
	if (templates === null) templates = buildTemplates();

	const ink = threshold(gray, w, h, Math.max(4, Math.round(Math.min(w, h) / 24)), INK_FIND);

	const quad = findGrid(gray, ink, w, h);
	if (quad === null) return null;

	const cell = cellPx || WARP_STILL;
	const size = cell * 9;
	const plane = {
		flat: rectify(gray, w, h, quad, size),
		size: size,
		radius: Math.round(cell / 2),
		scale: sideOf(quad) / size,
		quad: quad
	};

	// The cut is kept only if the lines come out more regular for it.
	const before = inkOf(plane, INK_BIAS);
	const bounds = gridBounds(before, size);
	if (bounds !== null) {
		const cropped = rectify(plane.flat, size, size, rectQuad(bounds), size);
		const after = threshold(cropped, size, size, plane.radius, INK_BIAS);
		if (bandScore(after, size) > bandScore(before, size) + 0.05) {
			plane.flat = cropped;
			plane.scale *= (bounds.y1 - bounds.y0) / size;
		}
	}
	return plane;
}

/**
 * Reads the digits off the straightened grid at one ink threshold. Returns
 * the grid, the ranking per cell and how clearly each cell was decided.
 */
function readCells(plane, bias) {
	const ink = inkOf(plane, bias);
	const thin = inkOf(plane, bias - THIN_DROP);
	const cells = cutCells(ink, thin, plane.size, plane.scale);

	const grid = S.emptyGrid();
	const margin = new Float64Array(S.CELLS);
	const faint = new Uint8Array(S.CELLS);
	const ranking = [];
	for (let i = 0; i < S.CELLS; i++) {
		if (cells[i] === null) { ranking.push(null); continue; }
		const ranked = classify(cells[i]);
		ranking.push(ranked);
		faint[i] = cells[i].faint ? 1 : 0;
		grid[i] = cells[i].faint ? 0 : ranked[0].digit;
		margin[i] = ranked[1].score === 0 ? 0 : (ranked[1].score - ranked[0].score) / ranked[1].score;
	}
	return {
		ok: true, grid: grid, ranking: ranking, margin: margin,
		faint: faint, quad: plane.quad
	};
}

/** Runs the whole chain on one frame at the usual threshold, dark ink assumed. */
function readImage(image, cellPx, bias) {
	const plane = straighten(toGray(image), image.width, image.height, cellPx);
	if (plane === null) return { ok: false, reason: "no grid found" };
	return readCells(plane, bias || INK_BIAS);
}

/* -------------------------------------------------------------------- Repair */

const FAINT_COST = 0.15;    // what it costs to read a digit into a nearly empty cell
const SURE_MARGIN = 0.12;   // below this a cell is shown as unconfirmed
const SINGLE_TRIES = 16;    // doubtful cells offered to the search, one at a time
const PAIR_TRIES = 10;      // and to the search over two of them

function unique(grid) {
	if (S.validate(grid).length > 0) return false;
	return S.countSolutions(grid, 2) === 1;
}

/**
 * The cells the reader was least sure of, each with the values it would still
 * accept. A cell with barely any ink counts as empty but keeps its digits, so
 * a missed given can be put back.
 */
function doubtfulCells(reading) {
	const list = [];
	for (let i = 0; i < S.CELLS; i++) {
		const ranked = reading.ranking[i];
		if (ranked === null) continue;
		const options = [];
		if (reading.faint[i] === 1) {
			options.push({ digit: 0, cost: 0 });
			for (let k = 0; k < 2; k++) {
				options.push({
					digit: ranked[k].digit,
					cost: FAINT_COST + ranked[k].score - ranked[0].score
				});
			}
		} else {
			for (let k = 0; k < 3; k++) {
				options.push({ digit: ranked[k].digit, cost: ranked[k].score - ranked[0].score });
			}
		}
		list.push({
			at: i,
			options: options,
			doubt: reading.faint[i] === 1 ? 1 : 1 - reading.margin[i]
		});
	}
	list.sort(function (a, b) { return b.doubt - a.doubt; });
	return list;
}

function swap(grid, cells, choice) {
	const out = grid.slice();
	const changed = [];
	let cost = 0;
	for (let k = 0; k < cells.length; k++) {
		const option = cells[k].options[choice[k]];
		out[cells[k].at] = option.digit;
		cost += option.cost;
		changed.push(cells[k].at);
	}
	return { grid: out, changed: changed, cost: cost };
}

/**
 * The cheapest way to make the grid unique by changing exactly depth of the
 * offered cells, each to one of the other values it would accept.
 */
function bestSwap(grid, cells, depth) {
	const chosen = [];
	const choice = [];
	let found = null;

	function walk(from) {
		if (chosen.length === depth) {
			const candidate = swap(grid, chosen, choice);
			if (!unique(candidate.grid)) return;
			if (found === null || candidate.cost < found.cost) found = candidate;
			return;
		}
		for (let i = from; i < cells.length; i++) {
			chosen.push(cells[i]);
			for (let k = 1; k < cells[i].options.length; k++) {
				choice.push(k);
				walk(i + 1);
				choice.pop();
			}
			chosen.pop();
		}
	}

	walk(0);
	return found;
}

/**
 * A correct reading has exactly one solution. If it has none or several, the
 * least certain cells are changed until it has.
 */
function repair(reading) {
	if (unique(reading.grid)) return { grid: reading.grid, changed: [] };

	const doubtful = doubtfulCells(reading);
	const single = bestSwap(reading.grid, doubtful.slice(0, SINGLE_TRIES), 1);
	if (single !== null) return single;
	return bestSwap(reading.grid, doubtful.slice(0, PAIR_TRIES), 2);
}

/** Marks the cells the user should look at before solving. */
function uncertainCells(reading, fixed) {
	const flags = new Uint8Array(S.CELLS);
	for (let i = 0; i < S.CELLS; i++) {
		if (reading.ranking[i] === null) continue;
		if (reading.faint[i] === 1 || reading.margin[i] < SURE_MARGIN) flags[i] = 1;
	}
	for (const i of fixed.changed) flags[i] = 1;
	return flags;
}

/**
 * Reads one straightened grid. A bright screen needs a softer threshold than
 * paper and a stained one a harder, so the thresholds are tried in turn until
 * the reading holds up.
 */
function readSolvable(plane) {
	let last = null;
	for (const bias of INK_TRIES) {
		const reading = readCells(plane, bias);
		const fixed = repair(reading);
		if (fixed !== null) {
			return {
				ok: true,
				grid: fixed.grid,
				changed: fixed.changed,
				uncertain: uncertainCells(reading, fixed),
				reading: reading
			};
		}
		last = reading;
	}
	return { ok: false, reason: "grid read, but it does not solve", reading: last };
}

/** Reads one frame at one polarity, dark ink on a light ground unless reversed. */
function readFrame(image, cellPx, dark) {
	const gray = toGray(image);
	const plane = straighten(dark ? invertPlane(gray) : gray,
		image.width, image.height, cellPx);
	if (plane === null) return { ok: false, reason: "no grid found" };
	return readSolvable(plane);
}

/**
 * Reads one frame and returns a grid the solver accepts, or a reason. Dark ink
 * on a light ground is the usual case. A screen in dark mode turns it around,
 * and neither the outline nor the digits survive that, so the whole chain runs
 * a second time on the reversed plane.
 */
function readPuzzle(image, cellPx) {
	let last = null;
	for (const dark of [false, true]) {
		const result = readFrame(image, cellPx, dark);
		if (result.ok) return result;
		// The first reading that got as far as the digits explains the failure best.
		if (last === null && result.reading) last = result;
	}
	return last === null ? { ok: false, reason: "no grid found" } : last;
}

/* ------------------------------------------------------------------- Capture */

const LIVE_SIDE = 1024;     // frames from the camera are scaled to this
const STILL_SIDE = 1600;    // a picked photo is scaled to this

/** Draws a video frame or an image onto a canvas and returns its pixels. */
function pixelsOf(source, width, height, maxSide) {
	const scale = Math.min(1, maxSide / Math.max(width, height));
	const w = Math.max(1, Math.round(width * scale));
	const h = Math.max(1, Math.round(height * scale));
	const ctx = drawingContext(w, h);
	ctx.drawImage(source, 0, 0, w, h);
	return ctx.getImageData(0, 0, w, h);
}

function loadImage(file) {
	return new Promise(function (resolve, reject) {
		const url = URL.createObjectURL(file);
		const img = new Image();
		img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
		img.onerror = function () { URL.revokeObjectURL(url); reject(new Error("image not readable")); };
		img.src = url;
	});
}

/* ------------------------------------------------------------------- Display */

const AGREE_FRAMES = 3;     // live frames that must agree before accepting
const AGREE_WINDOW = 6;     // and the span of frames they may come from
const LIVE_PAUSE = 80;      // rest between two frames, on top of the reading itself
const PROBE_EVERY = 4;      // every so many live frames one is read the other way round

function buildOverlay() {
	const root = document.createElement("div");
	root.className = "shot";
	root.hidden = true;
	root.innerHTML =
		'<div class="shot-stage">' +
		'<video class="shot-video" playsinline muted></video>' +
		'<div class="shot-frame"></div>' +
		'</div>' +
		'<p class="shot-hint" role="status"></p>' +
		'<div class="controls">' +
		'<button type="button" class="shot-pick" hidden>Choose photo</button>' +
		'<button type="button" class="shot-close">Cancel</button>' +
		'</div>';
	document.body.appendChild(root);
	return root;
}

function capitalise(text) {
	return text.charAt(0).toUpperCase() + text.slice(1);
}

/* Writes to the status line even when app.js has not published its hook. */
function say(text, isError) {
	if (S.ui) {
		S.ui.setStatus(text, isError);
		return;
	}
	const status = document.getElementById("status");
	if (status !== null) {
		status.textContent = text;
		status.classList.toggle("error", Boolean(isError));
	}
}

function initPhoto() {
	const button = document.getElementById("photo");
	if (button === null) return;

	const overlay = buildOverlay();
	const stage = overlay.querySelector(".shot-stage");
	const video = overlay.querySelector(".shot-video");
	const hint = overlay.querySelector(".shot-hint");
	const pick = overlay.querySelector(".shot-pick");
	const close = overlay.querySelector(".shot-close");

	const file = document.createElement("input");
	file.type = "file";
	file.accept = "image/*";
	file.setAttribute("capture", "environment");
	file.hidden = true;
	overlay.appendChild(file);

	let stream = null;
	let timer = 0;
	let dark = false;
	let frames = 0;
	const recent = [];

	/*
	 * How often this reading turned up in the last frames. A single odd frame
	 * used to reset the count, which made the camera feel stuck.
	 */
	function timesSeen(key) {
		recent.push(key);
		if (recent.length > AGREE_WINDOW) recent.shift();
		if (key === null) return 0;
		let seen = 0;
		for (const past of recent) {
			if (past === key) seen++;
		}
		return seen;
	}

	function stop() {
		recent.length = 0;
		dark = false;
		frames = 0;
		if (timer !== 0) { clearTimeout(timer); timer = 0; }
		if (stream !== null) {
			for (const track of stream.getTracks()) track.stop();
			stream = null;
		}
		video.srcObject = null;
		overlay.hidden = true;
	}

	function accept(result, note) {
		stop();
		if (!S.ui) {
			say("The board is not ready. Reload the page.", true);
			return;
		}
		S.ui.setPuzzle(result.grid, result.uncertain);
		const open = result.uncertain.reduce(function (n, v) { return n + v; }, 0);
		const checked = open === 0 ? "" :
			` Check the ${open} marked cell${open === 1 ? "" : "s"}.`;
		say(`${S.givens(result.grid)} givens read ${note}.${checked}`, false);
	}

	function tick() {
		timer = 0;
		if (stream === null || video.videoWidth === 0) {
			timer = setTimeout(tick, LIVE_PAUSE);
			return;
		}
		const frame = pixelsOf(video, video.videoWidth, video.videoHeight, LIVE_SIDE);
		/*
		 * One polarity per frame keeps an empty viewfinder cheap. Every fourth
		 * frame tries the other one, and whichever carries a whole puzzle
		 * becomes the one the rest of the frames use.
		 */
		const probe = (frames++ % PROBE_EVERY) === PROBE_EVERY - 1;
		const turned = probe ? !dark : dark;
		const result = readFrame(frame, WARP_LIVE, turned);
		if (result.ok) dark = turned;
		// An empty probe says nothing about the polarity in use, so it is dropped.
		if (probe && !result.ok) {
			timer = setTimeout(tick, LIVE_PAUSE);
			return;
		}
		const seen = timesSeen(result.ok ? S.gridToString(result.grid) : null);
		if (seen >= AGREE_FRAMES) {
			accept(result, "from the camera");
			return;
		}
		hint.textContent = result.ok
			? `Grid found, hold still (${seen}/${AGREE_FRAMES})`
			: "Point the camera at the puzzle";
		timer = setTimeout(tick, LIVE_PAUSE);
	}

	/* Without a live picture the empty viewfinder is only in the way. */
	function noCamera(text) {
		stage.hidden = true;
		pick.hidden = false;
		hint.textContent = text;
	}

	async function open() {
		overlay.hidden = false;
		recent.length = 0;
		hint.textContent = "Point the camera at the puzzle";
		stage.hidden = false;
		pick.hidden = true;
		if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
			noCamera("No camera here. Choose a photo instead.");
			return;
		}
		try {
			stream = await navigator.mediaDevices.getUserMedia({
				video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }
			});
			video.srcObject = stream;
			await video.play();
			tick();
		} catch (error) {
			noCamera("Camera not available. Choose a photo instead.");
		}
	}

	button.addEventListener("click", function () {
		open().catch(function (error) {
			overlay.hidden = false;
			hint.textContent = "Could not open the camera: " + error.message;
		});
	});
	close.addEventListener("click", stop);
	pick.addEventListener("click", function () { file.click(); });

	file.addEventListener("change", async function () {
		if (file.files.length === 0) return;
		hint.textContent = "Reading the photo";
		try {
			const img = await loadImage(file.files[0]);
			const result = readPuzzle(pixelsOf(img, img.naturalWidth, img.naturalHeight, STILL_SIDE));
			if (result.ok) {
				accept(result, "from the photo");
			} else {
				hint.textContent = capitalise(result.reason) + ". Try another photo.";
			}
		} catch (error) {
			hint.textContent = "That file is not an image.";
		}
		file.value = "";
	});
}

S.camera = {
	readImage: readImage, readFrame: readFrame, readPuzzle: readPuzzle, repair: repair
};

if (typeof document !== "undefined") {
	document.addEventListener("DOMContentLoaded", initPhoto);
}

})();
