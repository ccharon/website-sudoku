"use strict";

/*
 * Reads a Sudoku from a photo. The reading runs in WebAssembly, built from
 * crates/ocr: it finds the grid, straightens it, reads each cell with a small
 * neural network and lets the solver settle the doubtful cells. This file
 * takes the pictures and shows what came of them.
 *
 *   Reading           the WebAssembly reader, in the shape the display expects
 *   Capture           pixels out of the camera or a picked file
 *   Display           the viewfinder and what it reports
 */

(function () {

const S = globalThis.Sudoku;

/* ------------------------------------------------------------------- Reading */

const WARP_LIVE = 64;       // pixels per cell when reading the camera stream

/**
 * Unpacks a result of the reader and frees its memory in WebAssembly. The
 * net's own reading and the corners of the grid come along for the test pages.
 */
function shaped(read) {
	const out = { ok: read.ok };
	if (read.ok) {
		out.grid = read.grid;
		out.changed = Array.from(read.changed);
		out.uncertain = read.uncertain;
	} else {
		out.reason = read.reason;
	}
	const q = read.quad;
	if (read.readingGrid !== undefined && q !== undefined) {
		out.reading = { grid: read.readingGrid, quad: [[q[0], q[1]], [q[2], q[3]], [q[4], q[5]], [q[6], q[7]]] };
	}
	read.free();
	return out;
}

/** Reads one frame at one polarity, dark ink on a light ground unless reversed. */
function readFrame(image, cellPx, dark) {
	return shaped(S.reader.readFrame(image.data, image.width, image.height, cellPx, Boolean(dark)));
}

/** Reads one photo, dark ink on a light ground first and reversed after. */
function readPuzzle(image, cellPx) {
	return shaped(S.reader.readPuzzle(image.data, image.width, image.height, cellPx));
}

/* ------------------------------------------------------------------- Capture */

const LIVE_SIDE = 1024;     // frames from the camera are scaled to this
const STILL_SIDE = 1600;    // a picked photo is scaled to this

/** A canvas context of the given size, set up for reading the pixels back. */
function drawingContext(width, height) {
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	return canvas.getContext("2d", { willReadFrequently: true });
}

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
		/*
		 * A reading the solver had to mend contradicted itself, and the mending
		 * is a guess. Such a puzzle is shown but not solved, so the guess stays
		 * visible instead of disappearing into a solution.
		 */
		const mended = result.changed.length > 0;
		const read = S.ui.setPuzzle(result.grid, result.uncertain, !mended);
		if (read.conflict !== null) {
			say(read.conflict, true);
			return;
		}
		const open = result.uncertain.reduce(function (n, v) { return n + v; }, 0);
		const checked = open === 0 ? "" :
			` Check the ${open} marked cell${open === 1 ? "" : "s"}.`;
		const outcome = read.solutions !== 1
			? (read.solutions === 0 ? "but it has no solution" : "but it has several solutions")
			: (mended ? "and mended where it contradicted itself" : "and solved");
		say(`${S.givens(result.grid)} givens read ${note} ${outcome}.${checked}`,
			read.solutions !== 1);
	}

	/* One frame the reader chokes on must not put the viewfinder out. */
	function readSafely(frame, turned) {
		try {
			return readFrame(frame, WARP_LIVE, turned);
		} catch (error) {
			return { ok: false, reason: error.message };
		}
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
		const result = readSafely(frame, turned);
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
			hint.textContent = "Could not read that photo: " + error.message;
		}
		file.value = "";
	});
}

S.camera = { readFrame: readFrame, readPuzzle: readPuzzle };

if (typeof document !== "undefined") {
	document.addEventListener("DOMContentLoaded", initPhoto);
}

})();
