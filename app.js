"use strict";

/*
 * Sudoku in the browser. The solver turns the puzzle into an exact cover
 * problem and solves it with Knuth's Algorithm X as Dancing Links.
 */

const SIZE = 9;
const BOX_SIZE = 3;
const CELLS = SIZE * SIZE;

/* -------------------------------------------------------------------- Grid */

/** A grid is a Uint8Array of 81 cells, 0 for empty. */
function emptyGrid() {
	return new Uint8Array(CELLS);
}

function boxOf(row, col) {
	return Math.floor(row / BOX_SIZE) * BOX_SIZE + Math.floor(col / BOX_SIZE);
}

/**
 * Reads 81 cells. The characters '.', '0' and ' ' mean an empty cell, line
 * breaks and the frame characters '|', '-' and '+' are skipped.
 */
function parseGrid(text) {
	const grid = emptyGrid();
	let n = 0;
	for (const ch of text) {
		let value;
		if (ch >= "1" && ch <= "9") {
			value = ch.charCodeAt(0) - 48;
		} else if (ch === "." || ch === "0" || ch === " ") {
			value = 0;
		} else if (ch === "\n" || ch === "\r" || ch === "\t" || ch === "|" || ch === "-" || ch === "+") {
			continue;
		} else {
			throw new Error(`unexpected character ${JSON.stringify(ch)}`);
		}
		if (n === CELLS) {
			throw new Error("more than 81 cells");
		}
		grid[n++] = value;
	}
	if (n !== CELLS) {
		throw new Error(`expected 81 cells, read ${n}`);
	}
	return grid;
}

function gridToString(grid) {
	let out = "";
	for (let i = 0; i < CELLS; i++) {
		out += grid[i] === 0 ? "." : String(grid[i]);
	}
	return out;
}

function givens(grid) {
	let n = 0;
	for (let i = 0; i < CELLS; i++) {
		if (grid[i] !== 0) n++;
	}
	return n;
}

function isComplete(grid) {
	for (let i = 0; i < CELLS; i++) {
		if (grid[i] === 0) return false;
	}
	return true;
}

const UNIT_NAMES = ["row", "column", "box"];

/** Marks a cell whose value is not a digit from 1 to 9. */
const UNIT_VALUE = -1;

/**
 * Reports every cell outside the digit range and every unit that holds a digit
 * twice. Each affected cell is reported once, with the unit in the order row,
 * column, box.
 */
function validate(grid) {
	const rowFirst = new Int32Array(SIZE * (SIZE + 1)).fill(-1);
	const colFirst = new Int32Array(SIZE * (SIZE + 1)).fill(-1);
	const boxFirst = new Int32Array(SIZE * (SIZE + 1)).fill(-1);
	const conflicts = [];

	for (let i = 0; i < CELLS; i++) {
		const value = grid[i];
		if (value === 0) continue;
		const row = Math.floor(i / SIZE);
		const col = i % SIZE;
		if (!Number.isInteger(value) || value < 0 || value > SIZE) {
			conflicts.push({
				index: i, row: row, col: col, value: value, unit: UNIT_VALUE,
				otherIndex: i, otherRow: row, otherCol: col
			});
			continue;
		}
		const box = boxOf(row, col);
		const ri = row * (SIZE + 1) + value;
		const ci = col * (SIZE + 1) + value;
		const bi = box * (SIZE + 1) + value;

		let other = -1;
		let unit = 0;
		if (rowFirst[ri] >= 0) {
			other = rowFirst[ri];
			unit = 0;
		} else if (colFirst[ci] >= 0) {
			other = colFirst[ci];
			unit = 1;
		} else if (boxFirst[bi] >= 0) {
			other = boxFirst[bi];
			unit = 2;
		}
		if (other >= 0) {
			conflicts.push({
				index: i, row: row, col: col, value: value, unit: unit,
				otherIndex: other,
				otherRow: Math.floor(other / SIZE),
				otherCol: other % SIZE
			});
		}
		if (rowFirst[ri] < 0) rowFirst[ri] = i;
		if (colFirst[ci] < 0) colFirst[ci] = i;
		if (boxFirst[bi] < 0) boxFirst[bi] = i;
	}
	return conflicts;
}

function conflictText(c) {
	if (c.unit === UNIT_VALUE) {
		return `r${c.row + 1}c${c.col + 1} holds ${c.value}, not a digit from 1 to 9`;
	}
	return `Digit ${c.value} twice in ${UNIT_NAMES[c.unit]}: ` +
		`r${c.otherRow + 1}c${c.otherCol + 1}, r${c.row + 1}c${c.col + 1}`;
}

/* ------------------------------------------------------------- Exact Cover */

/*
 * 729 candidate rows of the form "digit d+1 sits in cell (r,c)" and 324
 * constraint columns in four groups of 81 each:
 *
 *   cell      0..80    r*9+c          cell (r,c) is filled
 *   row      81..161   81 + r*9 + d   digit d+1 occurs in row r
 *   column  162..242   162 + c*9 + d  digit d+1 occurs in column c
 *   box     243..323   243 + b*9 + d  digit d+1 occurs in box b
 */
const NUM_ROWS = CELLS * SIZE;
const NUM_COLS = 4 * CELLS;
const COL_CELL = 0;
const COL_ROW = CELLS;
const COL_COL = 2 * CELLS;
const COL_BOX = 3 * CELLS;

function rowIndex(row, col, digit) {
	return (row * SIZE + col) * SIZE + digit;
}

function colsOfRow(candidate) {
	const digit = candidate % SIZE;
	const cell = Math.floor(candidate / SIZE);
	const row = Math.floor(cell / SIZE);
	const col = cell % SIZE;
	return [
		COL_CELL + row * SIZE + col,
		COL_ROW + row * SIZE + digit,
		COL_COL + col * SIZE + digit,
		COL_BOX + boxOf(row, col) * SIZE + digit
	];
}

/* ------------------------------------------------------------ Dancing Links */

/*
 * Node layout: index 0 is the root, 1 to 324 are the column heads, then four
 * data nodes per candidate row.
 */
const ROOT = 0;
const NODE_BASE = 1 + NUM_COLS;
const NODES_PER_ROW = 4;
const TOTAL_NODES = NODE_BASE + NUM_ROWS * NODES_PER_ROW;

function headOf(col) {
	return col + 1;
}

/**
 * Builds the linked matrix. It is the same for every Sudoku and is left
 * unchanged after each run.
 */
function buildMatrix() {
	const m = {
		left: new Int32Array(TOTAL_NODES),
		right: new Int32Array(TOTAL_NODES),
		up: new Int32Array(TOTAL_NODES),
		down: new Int32Array(TOTAL_NODES),
		col: new Int32Array(TOTAL_NODES),
		row: new Int32Array(TOTAL_NODES),
		size: new Int32Array(NODE_BASE),
		sol: new Int32Array(CELLS),
		solLen: 0,
		random: null
	};

	let prev = ROOT;
	for (let c = 0; c < NUM_COLS; c++) {
		const h = headOf(c);
		m.left[h] = prev;
		m.right[prev] = h;
		m.up[h] = h;
		m.down[h] = h;
		m.col[h] = h;
		m.row[h] = -1;
		prev = h;
	}
	m.right[prev] = ROOT;
	m.left[ROOT] = prev;
	m.row[ROOT] = -1;
	m.col[ROOT] = ROOT;

	for (let r = 0; r < NUM_ROWS; r++) {
		const base = NODE_BASE + r * NODES_PER_ROW;
		const cols = colsOfRow(r);
		for (let k = 0; k < NODES_PER_ROW; k++) {
			const n = base + k;
			m.row[n] = r;
			m.left[n] = base + (k + NODES_PER_ROW - 1) % NODES_PER_ROW;
			m.right[n] = base + (k + 1) % NODES_PER_ROW;

			const h = headOf(cols[k]);
			const last = m.up[h];
			m.down[last] = n;
			m.up[n] = last;
			m.down[n] = h;
			m.up[h] = n;
			m.col[n] = h;
			m.size[h]++;
		}
	}
	return m;
}

/** Removes the column from the header row and every row that fills it. */
function cover(m, h) {
	m.right[m.left[h]] = m.right[h];
	m.left[m.right[h]] = m.left[h];
	for (let i = m.down[h]; i !== h; i = m.down[i]) {
		for (let j = m.right[i]; j !== i; j = m.right[j]) {
			m.down[m.up[j]] = m.down[j];
			m.up[m.down[j]] = m.up[j];
			m.size[m.col[j]]--;
		}
	}
}

/*
 * Undoes cover. The reverse order restores the state exactly, because every
 * removed node kept its neighbours.
 */
function uncover(m, h) {
	for (let i = m.up[h]; i !== h; i = m.up[i]) {
		for (let j = m.left[i]; j !== i; j = m.left[j]) {
			m.size[m.col[j]]++;
			m.down[m.up[j]] = j;
			m.up[m.down[j]] = j;
		}
	}
	m.right[m.left[h]] = h;
	m.left[m.right[h]] = h;
}

/**
 * Returns the head node of the constraint with the fewest remaining options,
 * or -1 once every constraint is satisfied.
 */
function chooseColumn(m) {
	let best = -1;
	let bestSize = Infinity;
	for (let h = m.right[ROOT]; h !== ROOT; h = m.right[h]) {
		if (m.size[h] < bestSize) {
			best = h;
			bestSize = m.size[h];
			if (bestSize <= 1) break;
		}
	}
	return best;
}

function selectRow(m, r) {
	const base = NODE_BASE + r * NODES_PER_ROW;
	for (let k = 0; k < NODES_PER_ROW; k++) {
		cover(m, m.col[base + k]);
	}
	m.sol[m.solLen++] = r;
}

function unselectRow(m, r) {
	const base = NODE_BASE + r * NODES_PER_ROW;
	for (let k = NODES_PER_ROW - 1; k >= 0; k--) {
		uncover(m, m.col[base + k]);
	}
	m.solLen--;
}

/**
 * Searches for assignments that satisfy every remaining constraint exactly
 * once. onSolution ends the search with true. The matrix is restored in any
 * case before this returns.
 */
function search(m, onSolution) {
	const h = chooseColumn(m);
	if (h < 0) {
		return onSolution(m.sol, m.solLen);
	}
	if (m.size[h] === 0) {
		return false;
	}

	cover(m, h);

	const rows = [];
	for (let i = m.down[h]; i !== h; i = m.down[i]) {
		rows.push(i);
	}
	if (m.random) {
		shuffle(rows, m.random);
	}

	for (const i of rows) {
		m.sol[m.solLen++] = m.row[i];
		for (let j = m.right[i]; j !== i; j = m.right[j]) {
			cover(m, m.col[j]);
		}

		const stop = search(m, onSolution);

		for (let j = m.left[i]; j !== i; j = m.left[j]) {
			uncover(m, m.col[j]);
		}
		m.solLen--;
		if (stop) {
			uncover(m, h);
			return true;
		}
	}
	uncover(m, h);
	return false;
}

/** The matrix is built once and reused across all runs. */
let sharedMatrix = null;

function matrix() {
	if (sharedMatrix === null) {
		sharedMatrix = buildMatrix();
	}
	return sharedMatrix;
}

/** Applies the givens, searches and restores the matrix. */
function run(grid, onSolution, random) {
	const conflicts = validate(grid);
	if (conflicts.length > 0) {
		throw new Error(conflictText(conflicts[0]));
	}
	const m = matrix();
	m.random = random || null;

	let applied = 0;
	try {
		for (let i = 0; i < CELLS; i++) {
			if (grid[i] === 0) continue;
			selectRow(m, rowIndex(Math.floor(i / SIZE), i % SIZE, grid[i] - 1));
			applied++;
		}
		search(m, onSolution);
		while (applied-- > 0) {
			unselectRow(m, m.sol[m.solLen - 1]);
		}
		m.random = null;
	} catch (error) {
		// A throw out of search leaves columns covered, so the matrix is dropped.
		sharedMatrix = null;
		throw error;
	}
}

function gridFromRows(sol, len) {
	const grid = emptyGrid();
	for (let k = 0; k < len; k++) {
		const candidate = sol[k];
		const digit = candidate % SIZE;
		const cell = Math.floor(candidate / SIZE);
		grid[cell] = digit + 1;
	}
	return grid;
}

/** Returns the first solution found, or null. */
function solve(grid, random) {
	let solution = null;
	run(grid, function (sol, len) {
		solution = gridFromRows(sol, len);
		return true;
	}, random);
	return solution;
}

/**
 * Counts the solutions and stops at limit. A limit of 2 is enough to check
 * uniqueness.
 */
function countSolutions(grid, limit) {
	const max = limit < 1 ? 1 : limit;
	let count = 0;
	run(grid, function () {
		count++;
		return count >= max;
	}, null);
	return count;
}

/* --------------------------------------------------------------- Generator */

/** Random numbers from a seed, so the same puzzle can be repeated. */
function randomFromSeed(seed) {
	let a = seed >>> 0;
	return function () {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function shuffle(items, random) {
	for (let i = items.length - 1; i > 0; i--) {
		const j = Math.floor(random() * (i + 1));
		const tmp = items[i];
		items[i] = items[j];
		items[j] = tmp;
	}
	return items;
}

/** Returns a random complete assignment. */
function fullGrid(random) {
	return solve(emptyGrid(), random);
}

/**
 * Returns a puzzle with exactly one solution. It starts from a complete
 * assignment and keeps removing cells as long as the solution stays unique.
 * With symmetric the cells are removed in point-symmetric pairs.
 */
function generate(seed, symmetric) {
	const random = randomFromSeed(seed);
	const grid = fullGrid(random);

	const order = shuffle(Array.from({ length: CELLS }, (_, i) => i), random);
	for (const i of order) {
		const cells = symmetric && i !== CELLS - 1 - i ? [i, CELLS - 1 - i] : [i];

		// A pair is visited twice, and the second time nothing is left in it.
		const saved = cells.map((c) => grid[c]);
		if (saved.every((v) => v === 0)) continue;
		cells.forEach((c) => { grid[c] = 0; });

		if (countSolutions(grid, 2) !== 1) {
			cells.forEach((c, k) => { grid[c] = saved[k]; });
		}
	}
	return grid;
}

/* -------------------------------------------------------------- Public API */

globalThis.Sudoku = {
	SIZE: SIZE,
	CELLS: CELLS,
	emptyGrid: emptyGrid,
	parseGrid: parseGrid,
	gridToString: gridToString,
	validate: validate,
	conflictText: conflictText,
	givens: givens,
	isComplete: isComplete,
	solve: solve,
	countSolutions: countSolutions,
	generate: generate,
	fullGrid: fullGrid,
	randomFromSeed: randomFromSeed
};

/* ----------------------------------------------------------------- Display */

function initUI() {
	const boardEl = document.getElementById("board");
	const statusEl = document.getElementById("status");
	const cells = [];

	// Cells that belong to the puzzle. What the solver adds is not listed here.
	const given = new Uint8Array(CELLS);

	const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

	for (let i = 0; i < CELLS; i++) {
		const input = document.createElement("input");
		input.className = "cell";
		input.type = "text";
		input.inputMode = "numeric";
		input.maxLength = 1;
		input.autocomplete = "off";
		input.setAttribute("aria-label", `r${Math.floor(i / SIZE) + 1}c${(i % SIZE) + 1}`);
		cells.push(input);
		boardEl.appendChild(input);
	}

	function setStatus(text, isError) {
		statusEl.textContent = text;
		statusEl.classList.toggle("error", Boolean(isError));
	}

	function readGrid() {
		const grid = emptyGrid();
		for (let i = 0; i < CELLS; i++) {
			const value = cells[i].value.trim();
			grid[i] = value === "" ? 0 : Number(value);
		}
		return grid;
	}

	/** Writes the grid into the fields. filled marks the added values. */
	function showGrid(grid, filled) {
		for (let i = 0; i < CELLS; i++) {
			cells[i].value = grid[i] === 0 ? "" : String(grid[i]);
			cells[i].classList.toggle("given", grid[i] !== 0 && given[i] === 1);
			cells[i].classList.toggle("filled", Boolean(filled) && grid[i] !== 0 && given[i] === 0);
			cells[i].classList.remove("conflict");
			cells[i].classList.remove("uncertain");
		}
	}

	function markConflicts(conflicts) {
		for (const c of conflicts) {
			cells[c.index].classList.add("conflict");
			cells[c.otherIndex].classList.add("conflict");
		}
	}

	function markUncertain(flags) {
		for (let i = 0; i < CELLS; i++) {
			cells[i].classList.toggle("uncertain", flags[i] === 1);
		}
	}

	function takeAsGiven(grid) {
		for (let i = 0; i < CELLS; i++) {
			given[i] = grid[i] !== 0 ? 1 : 0;
		}
	}

	function handleGenerate() {
		const seed = (Math.random() * 0x100000000) >>> 0;
		const started = performance.now();
		const grid = generate(seed, true);
		const elapsed = performance.now() - started;

		takeAsGiven(grid);
		showGrid(grid, false);
		setStatus(`${givens(grid)} givens, seed ${seed}, ${number.format(elapsed)} ms.`, false);
	}

	function handleCheck() {
		const grid = readGrid();
		takeAsGiven(grid);
		showGrid(grid, false);

		const conflicts = validate(grid);
		if (conflicts.length > 0) {
			markConflicts(conflicts);
			const more = conflicts.length > 1 ? ` (+${conflicts.length - 1} more)` : "";
			setStatus(conflictText(conflicts[0]) + more, true);
			return;
		}

		const count = countSolutions(grid, 2);
		if (count === 0) {
			setStatus("No conflicts, but no solution.", true);
		} else if (count === 1) {
			setStatus(`${givens(grid)} givens, one solution.`, false);
		} else {
			setStatus(`${givens(grid)} givens, several solutions.`, false);
		}
	}

	function handleSolve() {
		const grid = readGrid();
		takeAsGiven(grid);

		const conflicts = validate(grid);
		if (conflicts.length > 0) {
			showGrid(grid, false);
			markConflicts(conflicts);
			const more = conflicts.length > 1 ? ` (+${conflicts.length - 1} more)` : "";
			setStatus(conflictText(conflicts[0]) + more, true);
			return;
		}

		const started = performance.now();
		const solution = solve(grid, null);
		const elapsed = performance.now() - started;

		if (solution === null) {
			showGrid(grid, false);
			setStatus("No solution.", true);
			return;
		}
		showGrid(solution, true);
		setStatus(`Solved in ${number.format(elapsed)} ms.`, false);
	}

	/* The bin carries no label, so a first press on a filled board only warns. */
	let clearAsked = 0;

	function handleClear() {
		const now = Date.now();
		if (now - clearAsked > 4000 && givens(readGrid()) > 0) {
			clearAsked = now;
			setStatus("Press the bin again to clear the board.", false);
			return;
		}
		clearAsked = 0;
		given.fill(0);
		showGrid(emptyGrid(), false);
		setStatus("Enter digits, then solve.", false);
	}

	boardEl.addEventListener("input", function (event) {
		const i = cells.indexOf(event.target);
		if (i < 0) return;
		event.target.value = event.target.value.replace(/[^1-9]/g, "").slice(-1);
		event.target.classList.remove("conflict");
		event.target.classList.remove("uncertain");
		given[i] = event.target.value === "" ? 0 : 1;
		event.target.classList.toggle("given", given[i] === 1);
		event.target.classList.remove("filled");
		if (event.target.value !== "" && i + 1 < CELLS) {
			cells[i + 1].focus();
		}
	});

	boardEl.addEventListener("keydown", function (event) {
		const i = cells.indexOf(event.target);
		if (i < 0) return;
		const steps = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -SIZE, ArrowDown: SIZE };
		const step = steps[event.key];
		if (step !== undefined) {
			const next = i + step;
			if (next >= 0 && next < CELLS) {
				cells[next].focus();
				event.preventDefault();
			}
			return;
		}
		if (event.key === "Backspace" && event.target.value === "" && i > 0) {
			cells[i - 1].focus();
			event.preventDefault();
		}
	});

	boardEl.addEventListener("focusin", function (event) {
		if (event.target.classList.contains("cell")) {
			event.target.select();
		}
	});

	// Entry point for camera.js, which fills the board from a photo.
	globalThis.Sudoku.ui = {
		setStatus: setStatus,
		/**
		 * Takes a reading as the givens. With maySolve set, a reading free of
		 * conflicts and with one solution is solved at once. Returns the first
		 * conflict in the wording the buttons use, else the number of solutions
		 * counted to two, so the caller can say what became of the reading.
		 */
		setPuzzle: function (grid, uncertain, maySolve) {
			takeAsGiven(grid);
			const conflicts = validate(grid);
			if (conflicts.length > 0) {
				showGrid(grid, false);
				markConflicts(conflicts);
				if (uncertain) markUncertain(uncertain);
				const more = conflicts.length > 1 ? ` (+${conflicts.length - 1} more)` : "";
				return { solutions: 0, conflict: conflictText(conflicts[0]) + more };
			}
			const count = countSolutions(grid, 2);
			const solution = maySolve && count === 1 ? solve(grid, null) : null;
			showGrid(solution === null ? grid : solution, solution !== null);
			if (uncertain) markUncertain(uncertain);
			return { solutions: count, conflict: null };
		}
	};

	document.getElementById("generate").addEventListener("click", handleGenerate);
	document.getElementById("check").addEventListener("click", handleCheck);
	document.getElementById("solve").addEventListener("click", handleSolve);
	document.getElementById("clear").addEventListener("click", handleClear);
}

if (typeof document !== "undefined") {
	document.addEventListener("DOMContentLoaded", initUI);
}
