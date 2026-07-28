#!/usr/bin/env bash
set -euo pipefail

# Pinned browser-clangd toolchain. Gradle supplies both directories so the
# expensive LLVM build can survive `clean` while only generated web resources
# are copied into tidyparse-web/build.
: "${ROOT_DIR:?Gradle must set ROOT_DIR}"
: "${OUTPUT_DIR:?Gradle must set OUTPUT_DIR}"

EMSDK_VERSION="4.0.22"
EMSDK_COMMIT="15915cad554b707837024dc2758b6a1c5b94b036"
WASI_SDK_VERSION="29.0"
WASI_SDK_MAJOR="29"
WASI_SYSROOT_SHA256="d99d5c4b277a725b7b56bf9d591609972ecac3207520a66408927280f191f6c7"
LLVM_VERSION="21.1.0"
LLVM_MAJOR="21"
LLVM_COMMIT="3623fe661ae35c6c80ac221f14d85be76aa870f1"
CMAKE_VERSION="3.31.6"
NINJA_VERSION="1.11.1.4"
ARTIFACT_VERSION="llvm-21.1.0-emsdk-4.0.22-wasi-29.0-r2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p "$ROOT_DIR" "$OUTPUT_DIR"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd -P)"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

for command in bash git curl python3 tar shasum; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "clangd build requires '$command' on PATH" >&2
    exit 1
  fi
done

echo "clangd build workspace: $ROOT_DIR"

# Bootstrap pinned CMake/Ninja wheels locally. This keeps a clean macOS/Linux
# checkout from depending on Homebrew or the system package manager.
TOOLS_DIR="$ROOT_DIR/tools"
if [[ ! -x "$TOOLS_DIR/bin/cmake" || ! -x "$TOOLS_DIR/bin/ninja" ]]; then
  python3 -m venv "$TOOLS_DIR"
  "$TOOLS_DIR/bin/python" -m pip install --disable-pip-version-check \
    "cmake==$CMAKE_VERSION" \
    "ninja==$NINJA_VERSION"
fi
export PATH="$TOOLS_DIR/bin:$PATH"

if [[ "$("$TOOLS_DIR/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("cmake"))')" != "$CMAKE_VERSION" ]]; then
  echo "Unexpected CMake version in $TOOLS_DIR" >&2
  exit 1
fi
if [[ "$("$TOOLS_DIR/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("ninja"))')" != "$NINJA_VERSION" ]]; then
  echo "Unexpected Ninja version in $TOOLS_DIR" >&2
  exit 1
fi

clone_pinned() {
  local repository="$1"
  local tag="$2"
  local commit="$3"
  local destination="$4"

  if [[ ! -d "$destination/.git" ]]; then
    if [[ -e "$destination" ]]; then
      echo "Refusing to replace incomplete checkout: $destination" >&2
      exit 1
    fi
    git clone --branch "$tag" --depth 1 "$repository" "$destination"
  fi

  local actual
  actual="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual" != "$commit" ]]; then
    echo "Pinned checkout mismatch in $destination" >&2
    echo "expected $commit, found $actual" >&2
    exit 1
  fi
}

EMSDK_DIR="$ROOT_DIR/emsdk"
clone_pinned \
  "https://github.com/emscripten-core/emsdk.git" \
  "$EMSDK_VERSION" \
  "$EMSDK_COMMIT" \
  "$EMSDK_DIR"

if [[ ! -x "$EMSDK_DIR/upstream/emscripten/emcc" ]]; then
  "$EMSDK_DIR/emsdk" install "$EMSDK_VERSION"
fi
"$EMSDK_DIR/emsdk" activate "$EMSDK_VERSION"
# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null

WASI_SYSROOT="$ROOT_DIR/wasi-sysroot-$WASI_SDK_VERSION"
if [[ ! -f "$WASI_SYSROOT/.tidyparse-complete" ]]; then
  if [[ -e "$WASI_SYSROOT" ]]; then
    echo "Refusing to replace incomplete WASI sysroot: $WASI_SYSROOT" >&2
    exit 1
  fi

  DOWNLOAD_DIR="$ROOT_DIR/downloads"
  mkdir -p "$DOWNLOAD_DIR"
  WASI_ARCHIVE="$DOWNLOAD_DIR/wasi-sysroot-$WASI_SDK_VERSION.tar.gz"
  if [[ ! -f "$WASI_ARCHIVE" ]]; then
    curl --fail --location --retry 3 \
      "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-$WASI_SDK_MAJOR/wasi-sysroot-$WASI_SDK_VERSION.tar.gz" \
      --output "$WASI_ARCHIVE.part"
    mv "$WASI_ARCHIVE.part" "$WASI_ARCHIVE"
  fi
  printf '%s  %s\n' "$WASI_SYSROOT_SHA256" "$WASI_ARCHIVE" | shasum -a 256 --check

  EXTRACT_DIR="$(mktemp -d "$ROOT_DIR/.wasi-sysroot.XXXXXX")"
  trap 'rm -rf "$EXTRACT_DIR"' EXIT
  tar -xzf "$WASI_ARCHIVE" -C "$EXTRACT_DIR"
  mv "$EXTRACT_DIR/wasi-sysroot-$WASI_SDK_VERSION" "$WASI_SYSROOT"
  touch "$WASI_SYSROOT/.tidyparse-complete"
  rmdir "$EXTRACT_DIR"
  trap - EXIT
fi

LLVM_DIR="$ROOT_DIR/llvm-project"
clone_pinned \
  "https://github.com/llvm/llvm-project.git" \
  "llvmorg-$LLVM_VERSION" \
  "$LLVM_COMMIT" \
  "$LLVM_DIR"

if git -C "$LLVM_DIR" apply --reverse --check "$SCRIPT_DIR/wait_stdin.patch" 2>/dev/null; then
  :
elif git -C "$LLVM_DIR" apply --check "$SCRIPT_DIR/wait_stdin.patch"; then
  git -C "$LLVM_DIR" apply "$SCRIPT_DIR/wait_stdin.patch"
else
  echo "The clangd stdin patch does not apply cleanly to LLVM $LLVM_VERSION" >&2
  exit 1
fi

NATIVE_BUILD="$ROOT_DIR/build-native"
cmake -G Ninja -S "$LLVM_DIR/llvm" -B "$NATIVE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=clang
cmake --build "$NATIVE_BUILD" --target llvm-tblgen clang-tblgen

WASM_BUILD="$ROOT_DIR/build-wasm"
COMMON_CMAKE_ARGS=(
  -G Ninja
  -S "$LLVM_DIR/llvm"
  -B "$WASM_BUILD"
  -DCMAKE_CXX_FLAGS=-pthread\ -Dwait4=__syscall_wait4
  -DCMAKE_BUILD_TYPE=MinSizeRel
  -DLLVM_TARGET_ARCH=wasm32-emscripten
  -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-wasi
  -DLLVM_TARGETS_TO_BUILD=WebAssembly
  -DLLVM_ENABLE_PROJECTS=clang\;clang-tools-extra
  -DLLVM_TABLEGEN="$NATIVE_BUILD/bin/llvm-tblgen"
  -DCLANG_TABLEGEN="$NATIVE_BUILD/bin/clang-tblgen"
  -DLLVM_BUILD_STATIC=ON
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DLLVM_ENABLE_BACKTRACES=OFF
  -DLLVM_ENABLE_UNWIND_TABLES=OFF
  -DLLVM_ENABLE_CRASH_OVERRIDES=OFF
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF
  -DCLANGD_TIDY_CHECKS=OFF
  -DLLVM_ENABLE_TERMINFO=OFF
  -DLLVM_ENABLE_PIC=OFF
  -DLLVM_ENABLE_ZLIB=OFF
  -DCLANG_ENABLE_ARCMT=OFF
)

# Build the two upstream resource-header groups needed by the wasm32 target.
# Keep the downloaded WASI SDK pristine and assemble a disposable browser
# sysroot containing only those headers and the one target tree clangd uses.
emcmake cmake "${COMMON_CMAKE_ARGS[@]}" \
  -DCMAKE_EXE_LINKER_FLAGS=-pthread\ -s\ ENVIRONMENT=worker\ -s\ NO_INVOKE_RUN
cmake --build "$WASM_BUILD" \
  --target core-resource-headers webassembly-resource-headers

BROWSER_SYSROOT="$(mktemp -d "$ROOT_DIR/.browser-sysroot.XXXXXX")"
cleanup_browser_sysroot() {
  if [[ -n "${BROWSER_SYSROOT:-}" && -d "$BROWSER_SYSROOT" ]]; then
    cmake -E remove_directory "$BROWSER_SYSROOT"
  fi
}
trap cleanup_browser_sysroot EXIT

BROWSER_INCLUDE_DIR="$BROWSER_SYSROOT/include"
CLANG_RESOURCE_STAGE="$BROWSER_SYSROOT/clang-resource"
mkdir -p "$BROWSER_INCLUDE_DIR"
cmake -E copy_directory \
  "$WASI_SYSROOT/include/wasm32-wasi" \
  "$BROWSER_INCLUDE_DIR/wasm32-wasi"
cmake --install "$WASM_BUILD" \
  --prefix "$CLANG_RESOURCE_STAGE" \
  --component core-resource-headers
cmake --install "$WASM_BUILD" \
  --prefix "$CLANG_RESOURCE_STAGE" \
  --component webassembly-resource-headers
cmake -E copy_directory \
  "$CLANG_RESOURCE_STAGE/lib/clang/$LLVM_MAJOR/include" \
  "$BROWSER_INCLUDE_DIR"

for required_header in \
  "stddef.h" \
  "wasm_simd128.h" \
  "wasm32-wasi/stdio.h" \
  "wasm32-wasi/c++/v1/vector"; do
  if [[ ! -f "$BROWSER_INCLUDE_DIR/$required_header" ]]; then
    echo "Browser sysroot is missing required header: $required_header" >&2
    exit 1
  fi
done
for excluded_target in \
  "wasm32-wasi-threads" \
  "wasm32-wasip1" \
  "wasm32-wasip1-threads" \
  "wasm32-wasip2"; do
  if [[ -e "$BROWSER_INCLUDE_DIR/$excluded_target" ]]; then
    echo "Browser sysroot unexpectedly contains: $excluded_target" >&2
    exit 1
  fi
done

FINAL_LINKER_FLAGS="-pthread -s ENVIRONMENT=worker -s NO_INVOKE_RUN -s EXIT_RUNTIME -s INITIAL_MEMORY=2GB -s ALLOW_MEMORY_GROWTH -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=256kB -s EXPORTED_RUNTIME_METHODS=FS,callMain -s MODULARIZE -s EXPORT_ES6 -s WASM_BIGINT -s ASYNCIFY -s PTHREAD_POOL_SIZE=4 --embed-file=$BROWSER_INCLUDE_DIR@/usr/include"
emcmake cmake "${COMMON_CMAKE_ARGS[@]}" \
  "-DCMAKE_EXE_LINKER_FLAGS=$FINAL_LINKER_FLAGS"
cmake --build "$WASM_BUILD" --target clangd

for artifact in clangd.js clangd.wasm; do
  source_file="$WASM_BUILD/bin/$artifact"
  if [[ ! -s "$source_file" ]]; then
    echo "Missing clangd build output: $source_file" >&2
    exit 1
  fi
  cp "$source_file" "$OUTPUT_DIR/$artifact.tmp"
done

JS_SHA256="$(shasum -a 256 "$OUTPUT_DIR/clangd.js.tmp" | awk '{print $1}')"
WASM_SHA256="$(shasum -a 256 "$OUTPUT_DIR/clangd.wasm.tmp" | awk '{print $1}')"
JS_SIZE="$(wc -c < "$OUTPUT_DIR/clangd.js.tmp" | tr -d ' ')"
WASM_SIZE="$(wc -c < "$OUTPUT_DIR/clangd.wasm.tmp" | tr -d ' ')"
PATCH_SHA256="$(shasum -a 256 "$SCRIPT_DIR/wait_stdin.patch" | awk '{print $1}')"

cat > "$OUTPUT_DIR/clangd-manifest.json.tmp" <<EOF
{
  "artifactVersion": "$ARTIFACT_VERSION",
  "target": "wasm32-emscripten",
  "clangdAsyncThreads": 4,
  "llvm": { "version": "$LLVM_VERSION", "commit": "$LLVM_COMMIT" },
  "emsdk": { "version": "$EMSDK_VERSION", "commit": "$EMSDK_COMMIT" },
  "wasiSysroot": { "version": "$WASI_SDK_VERSION", "sha256": "$WASI_SYSROOT_SHA256" },
  "stdinPatchSha256": "$PATCH_SHA256",
  "artifacts": {
    "clangd.js": { "bytes": $JS_SIZE, "sha256": "$JS_SHA256" },
    "clangd.wasm": { "bytes": $WASM_SIZE, "sha256": "$WASM_SHA256" }
  }
}
EOF

mv "$OUTPUT_DIR/clangd.js.tmp" "$OUTPUT_DIR/clangd.js"
mv "$OUTPUT_DIR/clangd.wasm.tmp" "$OUTPUT_DIR/clangd.wasm"
mv "$OUTPUT_DIR/clangd-manifest.json.tmp" "$OUTPUT_DIR/clangd-manifest.json"

echo "Built $ARTIFACT_VERSION:"
echo "  $OUTPUT_DIR/clangd.js ($JS_SIZE bytes)"
echo "  $OUTPUT_DIR/clangd.wasm ($WASM_SIZE bytes)"
