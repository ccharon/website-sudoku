"use strict";

/*
 * Reads a Sudoku from a photo. The image is thresholded, the grid frame is
 * located and rectified to a square, every cell is normalised, and the digits
 * are matched against templates rendered from the browser's own fonts. The
 * solver then checks the reading and repairs the least certain cells.
 */

(function () {

const S = globalThis.Sudoku;

const WARP_LIVE = 64;       // pixels per cell when reading the camera stream
const WARP_STILL = 80;      // pixels per cell for a single picture
const NORM = 32;            // side of a normalised digit bitmap
const NORM_FIT = 24;        // the digit is scaled to fit this box
const CELL_INSET = 0.14;    // share of a cell dropped on each side
const LIVE_SIDE = 1024;     // frames from the camera are scaled to this
const STILL_SIDE = 1600;    // a picked photo is scaled to this
const SURE_MARGIN = 0.12;   // below this a cell is shown as unconfirmed
const AGREE_FRAMES = 3;     // live frames that must agree before accepting
const HOLE_COST = 0.2;      // penalty per enclosed area two shapes differ by

/* ------------------------------------------------------------- Grey and ink */

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

/* --------------------------------------------------------------- Components */

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
		const box = { id: boxes.length, count: 0, x0: w, y0: h, x1: -1, y1: -1 };
		let top = 0;
		stack[top++] = start;
		labels[start] = box.id;
		while (top > 0) {
			const p = stack[--top];
			const x = p % w;
			const y = (p - x) / w;
			box.count++;
			if (x < box.x0) box.x0 = x;
			if (x > box.x1) box.x1 = x;
			if (y < box.y0) box.y0 = y;
			if (y > box.y1) box.y1 = y;
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

/* ----------------------------------------------------------- Grid detection */

/**
 * Picks the ink patch that looks like the grid frame: wide, roughly square and
 * an outline rather than a solid area. Returns the four corners in the order
 * top left, top right, bottom right, bottom left.
 */
function findGrid(mask, w, h) {
	const found = components(mask, w, h);
	const minArea = w * h * 0.04;
	let best = null;
	for (const box of found.boxes) {
		const bw = box.x1 - box.x0 + 1;
		const bh = box.y1 - box.y0 + 1;
		const area = bw * bh;
		if (area < minArea) continue;
		if (bw / bh < 0.5 || bw / bh > 2) continue;
		const fill = box.count / area;
		if (fill < 0.02 || fill > 0.6) continue;
		if (best === null || area > best.area) best = { box: box, area: area };
	}
	if (best === null) return null;

	const id = best.box.id;
	let tl = null, tr = null, br = null, bl = null;
	let tlV = Infinity, brV = -Infinity, trV = -Infinity, blV = Infinity;
	for (let y = best.box.y0; y <= best.box.y1; y++) {
		for (let x = best.box.x0; x <= best.box.x1; x++) {
			if (found.labels[y * w + x] !== id) continue;
			const sum = x + y;
			const diff = x - y;
			if (sum < tlV) { tlV = sum; tl = [x, y]; }
			if (sum > brV) { brV = sum; br = [x, y]; }
			if (diff > trV) { trV = diff; tr = [x, y]; }
			if (diff < blV) { blV = diff; bl = [x, y]; }
		}
	}
	const quad = [tl, tr, br, bl];
	return plausibleQuad(quad) ? quad : null;
}

/** Rejects quadrilaterals too lopsided to be a photographed grid. */
function plausibleQuad(quad) {
	let shortest = Infinity;
	let longest = 0;
	for (let i = 0; i < 4; i++) {
		const a = quad[i];
		const b = quad[(i + 1) % 4];
		const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
		if (len < shortest) shortest = len;
		if (len > longest) longest = len;
	}
	return shortest > 8 && longest / shortest < 2.5;
}

/* ---------------------------------------------------------------- Rectify */

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

/* ------------------------------------------------------------------ Cells */

/**
 * Scales the digit to a fixed box and centres it by its centre of mass, so
 * font size and position inside the cell stop mattering.
 */
function normalise(bitmap, w, box) {
	const bw = box.x1 - box.x0 + 1;
	const bh = box.y1 - box.y0 + 1;
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
function isolateDigit(mask, size, x0, y0, x1, y1) {
	const w = x1 - x0;
	const h = y1 - y0;
	const sub = new Uint8Array(w * h);
	let ink = 0;
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const v = mask[(y0 + y) * size + x0 + x];
			sub[y * w + x] = v;
			ink += v;
		}
	}
	if (ink < w * h * 0.02) return null;

	const found = components(sub, w, h);
	let best = null;
	for (const box of found.boxes) {
		const bw = box.x1 - box.x0 + 1;
		const bh = box.y1 - box.y0 + 1;
		if (box.count < 10) continue;
		if (bh < h * 0.3) continue;      // flat leftovers are grid lines
		if (bw > w * 0.92) continue;     // a run across the cell is a line
		if (best === null || box.count > best.count) best = box;
	}
	if (best === null) return null;

	const digit = new Uint8Array(w * h);
	for (let p = 0; p < w * h; p++) digit[p] = found.labels[p] === best.id ? 1 : 0;
	const norm = normalise(digit, w, best);
	if (norm === null) return null;
	const bw = best.x1 - best.x0 + 1;
	const bh = best.y1 - best.y0 + 1;
	return { mask: norm, holes: holeCount(digit, w, h, minHole(bw, bh)) };
}

/**
 * Cuts the straightened grid into cells. The outer share of every cell is
 * dropped so the grid lines fall away.
 */
function cutCells(mask, size) {
	const cell = size / 9;
	const inset = Math.round(cell * CELL_INSET);
	const out = [];
	for (let r = 0; r < 9; r++) {
		for (let c = 0; c < 9; c++) {
			out.push(isolateDigit(mask, size,
				Math.round(c * cell) + inset, Math.round(r * cell) + inset,
				Math.round((c + 1) * cell) - inset, Math.round((r + 1) * cell) - inset));
		}
	}
	return out;
}

/* --------------------------------------------------------- Shape matching */

/** Chamfer distance to the nearest ink pixel, two passes over the bitmap. */
function distanceMap(bitmap) {
	const dist = new Float32Array(NORM * NORM);
	for (let i = 0; i < dist.length; i++) dist[i] = bitmap[i] ? 0 : 1e4;
	for (let y = 0; y < NORM; y++) {
		for (let x = 0; x < NORM; x++) {
			const i = y * NORM + x;
			let d = dist[i];
			if (y > 0) {
				if (dist[i - NORM] + 3 < d) d = dist[i - NORM] + 3;
				if (x > 0 && dist[i - NORM - 1] + 4 < d) d = dist[i - NORM - 1] + 4;
				if (x < NORM - 1 && dist[i - NORM + 1] + 4 < d) d = dist[i - NORM + 1] + 4;
			}
			if (x > 0 && dist[i - 1] + 3 < d) d = dist[i - 1] + 3;
			dist[i] = d;
		}
	}
	for (let y = NORM - 1; y >= 0; y--) {
		for (let x = NORM - 1; x >= 0; x--) {
			const i = y * NORM + x;
			let d = dist[i];
			if (y < NORM - 1) {
				if (dist[i + NORM] + 3 < d) d = dist[i + NORM] + 3;
				if (x > 0 && dist[i + NORM - 1] + 4 < d) d = dist[i + NORM - 1] + 4;
				if (x < NORM - 1 && dist[i + NORM + 1] + 4 < d) d = dist[i + NORM + 1] + 4;
			}
			if (x < NORM - 1 && dist[i + 1] + 3 < d) d = dist[i + 1] + 3;
			dist[i] = d;
		}
	}
	for (let i = 0; i < dist.length; i++) dist[i] /= 3;
	return dist;
}

/** A counter smaller than this share of the digit is threshold noise. */
function minHole(bw, bh) {
	return Math.max(2, Math.round(bw * bh * 0.004));
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

function shapeOf(mask, holes) {
	return { mask: mask, dist: distanceMap(mask), holes: holes };
}

/** Distance plus the difference in enclosed areas. Lower fits better. */
function shapeDistance(a, b) {
	return chamfer(a, b) + HOLE_COST * Math.abs(a.holes - b.holes);
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

/* ---------------------------------------------------------------- Templates */

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

/**
 * Draws the digits 1 to 9 in several fonts and normalises them like the cells.
 * The font rasteriser of the browser takes the place of a trained model.
 */
function buildTemplates() {
	const side = 160;
	const canvas = document.createElement("canvas");
	canvas.width = side;
	canvas.height = side;
	const ctx = canvas.getContext("2d", { willReadFrequently: true });
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
			const box = { x0: side, y0: side, x1: -1, y1: -1 };
			for (let y = 0; y < side; y++) {
				for (let x = 0; x < side; x++) {
					if (data[(y * side + x) * 4] >= 128) continue;
					bitmap[y * side + x] = 1;
					if (x < box.x0) box.x0 = x;
					if (x > box.x1) box.x1 = x;
					if (y < box.y0) box.y0 = y;
					if (y > box.y1) box.y1 = y;
				}
			}
			if (box.x1 < 0) continue;
			const mask = normalise(bitmap, side, box);
			if (mask === null) continue;
			const key = digit + ":" + mask.join("");
			if (seen.has(key)) continue;    // the same face under another name
			seen.add(key);
			const holes = holeCount(bitmap, side, side,
				minHole(box.x1 - box.x0 + 1, box.y1 - box.y0 + 1));
			const shape = shapeOf(mask, holes);
			shape.digit = digit;
			out.push(shape);
		}
	}
	return out;
}

/** Ranks the digits 1 to 9 for one isolated cell. */
function classify(cell) {
	const shape = shapeOf(cell.mask, cell.holes);
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

/* ------------------------------------------------------------------ Reading */

/**
 * Runs the whole chain on one frame. Returns the grid, the ranking per cell
 * and how clearly each cell was decided.
 */
function readImage(image, cellPx) {
	if (templates === null) templates = buildTemplates();

	const w = image.width;
	const h = image.height;
	const gray = toGray(image);
	const ink = threshold(gray, w, h, Math.max(4, Math.round(Math.min(w, h) / 24)), 0.88);

	const quad = findGrid(ink, w, h);
	if (quad === null) return { ok: false, reason: "no grid found" };

	const cell = cellPx || WARP_STILL;
	const size = cell * 9;
	const flat = rectify(gray, w, h, quad, size);
	const flatInk = threshold(flat, size, size, Math.round(cell / 2), 0.9);
	const cells = cutCells(flatInk, size);

	const grid = S.emptyGrid();
	const margin = new Float64Array(S.CELLS);
	const ranking = [];
	for (let i = 0; i < S.CELLS; i++) {
		if (cells[i] === null) { ranking.push(null); continue; }
		const ranked = classify(cells[i]);
		ranking.push(ranked);
		grid[i] = ranked[0].digit;
		margin[i] = ranked[1].score === 0 ? 0 : (ranked[1].score - ranked[0].score) / ranked[1].score;
	}
	return { ok: true, grid: grid, ranking: ranking, margin: margin, quad: quad };
}

/* ------------------------------------------------------------------- Repair */

function unique(grid) {
	if (S.validate(grid).length > 0) return false;
	return S.countSolutions(grid, 2) === 1;
}

function swap(reading, cells, choice) {
	const grid = reading.grid.slice();
	let cost = 0;
	for (let k = 0; k < cells.length; k++) {
		const ranked = reading.ranking[cells[k]];
		grid[cells[k]] = ranked[choice[k]].digit;
		cost += ranked[choice[k]].score - ranked[0].score;
	}
	return { grid: grid, changed: cells.slice(), cost: cost };
}

/**
 * A correct reading has exactly one solution. If it has none or several, the
 * least certain cells are swapped for their runners up until it has.
 */
function repair(reading) {
	if (unique(reading.grid)) return { grid: reading.grid, changed: [] };

	const order = [];
	for (let i = 0; i < S.CELLS; i++) {
		if (reading.ranking[i] !== null) order.push(i);
	}
	order.sort(function (a, b) { return reading.margin[a] - reading.margin[b]; });

	let found = null;
	for (const i of order.slice(0, 14)) {
		for (let k = 1; k < 3; k++) {
			const candidate = swap(reading, [i], [k]);
			if (!unique(candidate.grid)) continue;
			if (found === null || candidate.cost < found.cost) found = candidate;
		}
	}
	if (found !== null) return found;

	const pairs = order.slice(0, 9);
	for (let a = 0; a < pairs.length; a++) {
		for (let b = a + 1; b < pairs.length; b++) {
			for (let ka = 1; ka < 3; ka++) {
				for (let kb = 1; kb < 3; kb++) {
					const candidate = swap(reading, [pairs[a], pairs[b]], [ka, kb]);
					if (!unique(candidate.grid)) continue;
					if (found === null || candidate.cost < found.cost) found = candidate;
				}
			}
		}
	}
	return found;
}

/** Marks the cells the user should look at before solving. */
function uncertainCells(reading, fixed) {
	const flags = new Uint8Array(S.CELLS);
	for (let i = 0; i < S.CELLS; i++) {
		if (reading.ranking[i] !== null && reading.margin[i] < SURE_MARGIN) flags[i] = 1;
	}
	for (const i of fixed.changed) flags[i] = 1;
	return flags;
}

/** Reads one frame and returns a grid the solver accepts, or a reason. */
function readPuzzle(image, cellPx) {
	const reading = readImage(image, cellPx);
	if (!reading.ok) return reading;
	const fixed = repair(reading);
	if (fixed === null) {
		return { ok: false, reason: "grid read, but it does not solve", reading: reading };
	}
	return {
		ok: true,
		grid: fixed.grid,
		changed: fixed.changed,
		uncertain: uncertainCells(reading, fixed),
		reading: reading
	};
}

/* ---------------------------------------------------------------- Capture */

/** Draws a video frame or an image onto a canvas and returns its pixels. */
function pixelsOf(source, width, height, maxSide) {
	const scale = Math.min(1, maxSide / Math.max(width, height));
	const w = Math.max(1, Math.round(width * scale));
	const h = Math.max(1, Math.round(height * scale));
	const canvas = document.createElement("canvas");
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext("2d", { willReadFrequently: true });
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

/* --------------------------------------------------------------------- UI */

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
		'<button type="button" class="shot-take">Capture</button>' +
		'<button type="button" class="shot-pick">Choose photo</button>' +
		'<button type="button" class="shot-close">Cancel</button>' +
		'</div>';
	document.body.appendChild(root);
	return root;
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
	const take = overlay.querySelector(".shot-take");
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
	let agree = null;
	let agreed = 0;

	function stop() {
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
			timer = setTimeout(tick, 200);
			return;
		}
		const frame = pixelsOf(video, video.videoWidth, video.videoHeight, LIVE_SIDE);
		const result = readPuzzle(frame, WARP_LIVE);
		if (result.ok) {
			const key = S.gridToString(result.grid);
			agreed = key === agree ? agreed + 1 : 1;
			agree = key;
			hint.textContent = `Grid found, holding still (${agreed}/${AGREE_FRAMES})`;
			if (agreed >= AGREE_FRAMES) {
				accept(result, "from the camera");
				return;
			}
		} else {
			agree = null;
			agreed = 0;
			hint.textContent = "Point the camera at the puzzle";
		}
		timer = setTimeout(tick, 200);
	}

	/* Without a live picture the empty viewfinder is only in the way. */
	function noCamera(text) {
		stage.hidden = true;
		take.hidden = true;
		hint.textContent = text;
	}

	async function open() {
		overlay.hidden = false;
		agree = null;
		agreed = 0;
		hint.textContent = "Point the camera at the puzzle";
		stage.hidden = false;
		take.hidden = false;
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

	take.addEventListener("click", function () {
		if (stream === null || video.videoWidth === 0) return;
		const result = readPuzzle(pixelsOf(video, video.videoWidth, video.videoHeight, STILL_SIDE));
		if (result.ok) {
			accept(result, "from the camera");
		} else {
			hint.textContent = capitalise(result.reason) + ". Try again.";
		}
	});

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

function capitalise(text) {
	return text.charAt(0).toUpperCase() + text.slice(1);
}

S.camera = { readImage: readImage, readPuzzle: readPuzzle, repair: repair };

if (typeof document !== "undefined") {
	document.addEventListener("DOMContentLoaded", initPhoto);
}

})();
