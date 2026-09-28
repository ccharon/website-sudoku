#!/bin/bash
# Builds the site into a directory: the WebAssembly module, the scripts
# bundled and minified, the stylesheet minified, the version in index.html
# and compressed twins for gzip_static and brotli_static.
#
#   ./build.sh <out dir> [version]
#
# Needs node, cargo with the target wasm32-unknown-unknown, and the tools
# from package.json installed with npm ci.
set -euo pipefail

out=${1:?usage: ./build.sh <out dir> [version]}
version=${2:-dev}
root=$(cd "$(dirname "$0")" && pwd)
web=$root/web
PATH=$root/node_modules/.bin:$PATH

rm -rf "$out"
mkdir -p "$out/pkg" "$out/icons"

wasm-pack build "$root/crates/web" --release --target web --out-dir "$web/pkg" --no-pack
cp "$web/pkg/sudoku_web_bg.wasm" "$out/pkg/"

# app.js is a module and pulls in the glue from pkg/. camera.js stays a
# classic script, because it must run after app.js has set up Sudoku.
esbuild "$web/app.js" --bundle --format=esm --minify --log-level=warning --outfile="$out/app.js"
esbuild "$web/camera.js" --minify --log-level=warning --outfile="$out/camera.js"
lightningcss --minify "$web/style.css" -o "$out/style.css"

cp "$web/index.html" "$out/"
cp "$web/icons/favicon.svg" "$web/icons/favicon-32.png" "$web/icons/apple-touch-icon.png" "$out/icons/"
sed -i -E "s/(href|src)=\"(app\.js|camera\.js|style\.css)\"/\1=\"\2?v=$version\"/g" "$out/index.html"
sed -i -E "s|(<span id=\"build\">)[^<]*(</span>)|\1$version\2|" "$out/index.html"

# Twins for gzip_static and brotli_static, written once index.html is final.
cd "$out"
node -e '
  const fs = require("fs"), zlib = require("zlib");
  for (const f of process.argv.slice(1)) {
    const raw = fs.readFileSync(f);
    fs.writeFileSync(f + ".gz", zlib.gzipSync(raw, { level: 9 }));
    fs.writeFileSync(f + ".br", zlib.brotliCompressSync(raw, { params: {
      [zlib.constants.BROTLI_PARAM_QUALITY]: 11,
      [zlib.constants.BROTLI_PARAM_SIZE_HINT]: raw.length
    } }));
    console.log(f, raw.length, fs.statSync(f + ".gz").size, fs.statSync(f + ".br").size);
  }
' index.html style.css app.js camera.js pkg/sudoku_web_bg.wasm icons/favicon.svg
