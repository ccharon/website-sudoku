"use strict";

/*
 * Sudoku in the browser. Solver and photo reader run as WebAssembly, built
 * from crates/, and this module draws the board around them.
 */

import init, { validate, solve, countSolutions, generate, readPuzzle, readFrame } from "./pkg/sudoku_web.js";

const SIZE = 9;
const CELLS = SIZE * SIZE;

/* -------------------------------------------------------------------- Grid */

/** A grid is a Uint8Array of 81 cells, 0 for empty. */
function emptyGrid() {
	return new Uint8Array(CELLS);
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

/** The first conflict in words, with a count of the others. */
function conflictSummary(conflicts) {
	const more = conflicts.length > 1 ? ` (+${conflicts.length - 1} more)` : "";
	return conflicts[0].text + more;
}

/* -------------------------------------------------------------- Public API */

// Set before init waits for WebAssembly, because camera.js reads it as soon as it runs.
globalThis.Sudoku = {
	SIZE: SIZE,
	CELLS: CELLS,
	emptyGrid: emptyGrid,
	gridToString: gridToString,
	givens: givens,
	validate: validate,
	solve: solve,
	countSolutions: countSolutions,
	generate: generate,
	// camera.js reads photos with these.
	reader: { readPuzzle: readPuzzle, readFrame: readFrame }
};

/* ----------------------------------------------------------------- Display */

const SEED_BOUND = 0x100000000;    // seeds are the 32 bit numbers below this
const CLEAR_ASK_MS = 4000;         // a warned bin stays armed this long

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
			cells[i].classList.remove("conflict", "uncertain");
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

	/** Takes what stands on the board as the puzzle and checks it over. */
	function readBoard() {
		const grid = readGrid();
		takeAsGiven(grid);
		return { grid: grid, conflicts: validate(grid) };
	}

	/** Puts the reading back on the board and points at the first conflict. */
	function showConflicts(grid, conflicts) {
		showGrid(grid, false);
		markConflicts(conflicts);
		setStatus(conflictSummary(conflicts), true);
	}

	function handleGenerate() {
		const seed = (Math.random() * SEED_BOUND) >>> 0;
		const started = performance.now();
		const grid = generate(seed, true);
		const elapsed = performance.now() - started;

		takeAsGiven(grid);
		showGrid(grid, false);
		setStatus(`${givens(grid)} givens, seed ${seed}, ${number.format(elapsed)} ms.`, false);
	}

	function handleCheck() {
		const board = readBoard();
		if (board.conflicts.length > 0) {
			showConflicts(board.grid, board.conflicts);
			return;
		}
		showGrid(board.grid, false);

		const count = countSolutions(board.grid, 2);
		if (count === 0) {
			setStatus("No conflicts, but no solution.", true);
		} else if (count === 1) {
			setStatus(`${givens(board.grid)} givens, one solution.`, false);
		} else {
			setStatus(`${givens(board.grid)} givens, several solutions.`, false);
		}
	}

	function handleSolve() {
		const board = readBoard();
		if (board.conflicts.length > 0) {
			showConflicts(board.grid, board.conflicts);
			return;
		}

		const started = performance.now();
		const solution = solve(board.grid);
		const elapsed = performance.now() - started;

		if (solution === undefined) {
			showGrid(board.grid, false);
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
		if (now - clearAsked > CLEAR_ASK_MS && givens(readGrid()) > 0) {
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
		event.target.classList.remove("conflict", "uncertain", "filled");
		given[i] = event.target.value === "" ? 0 : 1;
		event.target.classList.toggle("given", given[i] === 1);
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
				return { solutions: 0, conflict: conflictSummary(conflicts) };
			}
			const count = countSolutions(grid, 2);
			const solution = maySolve && count === 1 ? solve(grid) : undefined;
			showGrid(solution === undefined ? grid : solution, solution !== undefined);
			if (uncertain) markUncertain(uncertain);
			return { solutions: count, conflict: null };
		}
	};

	document.getElementById("generate").addEventListener("click", handleGenerate);
	document.getElementById("check").addEventListener("click", handleCheck);
	document.getElementById("solve").addEventListener("click", handleSolve);
	document.getElementById("clear").addEventListener("click", handleClear);
}

// Relative to this file, bundled or not. The version query of app.js carries
// over, so a new build fetches the module anew.
await init({ module_or_path: new URL("pkg/sudoku_web_bg.wasm" + new URL(import.meta.url).search, import.meta.url) });
if (typeof document !== "undefined") {
	initUI();
}
