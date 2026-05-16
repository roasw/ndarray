# AGENTS.md

## Scope

- `ndarray`: C++ column-major DLPack container with Python interop and zero-copy.
- Upsampling pipeline: Python export -> AOTI `.pt2` -> C++ runtime.
- C++ torch CPU kernels: under `src/kernel`, registered to torch and usable by AOT export/compile.

## Mandatory workflow

- Always ask for confirmation before applying code changes.
- Build/test after edits:
  - `cmake --build build/Debug`
  - `ctest --test-dir build/Debug --output-on-failure -LE benchmark`
- Run linters before commit: `pre-commit run --all`.
- Keep zero-copy behavior as a hard requirement for ndarray/torch DLPack interop.

## Project layout

- Headers: `inc/container`, `inc/algorithm`.
- Sources: `src/container`, `src/algorithm`, `src/kernel`.
- Python: package `python/ndarry`.
- Tests: `tests/demo`, `tests/ndarray`, `tests/algorithm`, `tests/python`.
- Benchmarks: `benchmark/`.

## Benchmarks

- Benchmarks and profiling tests are registered in CTest but excluded by default
  via labels. Use `-LE benchmark` for normal test runs.
- Run benchmarks: `ctest --test-dir build/Release -L benchmark`.
- Results are documented in `doc/benchmark-upsample-fourier.md`.

### Profiling

- Profiling is a CTest with label `profile`, off by default.
- The trace is written to `build/<Preset>/profile_upsample_2d_fourier.json`.
- Run: `ctest --test-dir build/Release -L profile`.
- View the trace: open `chrome://tracing` and load the JSON, or use `speedscope`.
- Quick CLI analysis with `jq`:
  ```bash
  # Top ops by total time
  jq '[.traceEvents[] | select(.ph=="X" and .dur) | {name:(.cat+"::"+.name), dur_us:.dur}] |
    group_by(.name) | map({name:.[0].name, total_ms:([.[].dur_us]|add/1000), calls:length}) |
    sort_by(-.total_ms) | .[0:15][] | "\(.total_ms|tostring|.[0:8])\t\(.calls)x\t\(.name)"' \
    build/Release/profile_upsample_2d_fourier.json
  ```

## Documentation

- Doxygen docs are generated from public headers and sources listed in `Doxyfile.in`.
- Build docs target (excluded from default `all`):
  - `cmake --build build/Debug --target doc`
- Generated HTML entrypoint:
  - `build/Debug/docs/doxygen/html/index.html`

## Naming contract for export/runtime

- Keep algorithm basenames aligned across Python, CMake metadata, and C++ runtime.
- In CMake, prefer `ALGORITHM_FILE`; `ALGORITHM_MODULE` is derived from that file path.
- The Python export pipeline derives `algorithm_name` from module basename (which tracks the Python filename stem when conventions are followed).
- Exported `model_name` entries must be prefixed with `<algorithm_name>_`.
- Metadata basename must match `algorithm_name`.
- C++ runtime intentionally relies on `__FILE__` stem conventions for matching.
- If you rename an algorithm, rename both Python and C++ algorithm files and keep prefixes in sync.

## Kernel constraints (torch CPU kernel)

- Register custom ops with torch dispatcher (`TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`).
- Keep kernel CPU-only for now; reserve naming for future CUDA variants.
- Ensure Python export path can call the registered op (so `torch.export` and AOTI can compile it).
- Add parity tests against current algorithm behavior and C++ runtime coverage.
- Do not introduce copies across ndarray\<->torch bridges unless explicitly required.

## Style & conventions

- C++: existing formatting patterns, `#pragma once`, methods `UpperCamelCase`.
- C++ namespaces use `UpperCamelCase`.
- Private class members use `m_` prefix; struct/public fields use `lowerCamelCase`.
- C++ local variables and function parameters use `lowerCamelCase` (no `snake_case`).
- Header include order: STL, blank, third-party, blank, project.
- Keep public headers declaration-focused and readable.
- `ndarray` remains templated with explicit instantiation in `.cpp`.

## Commits

- Message format: `(chore|doc|fix|feat|infra|refac|revert): <description>.`
- Max 72 chars, must end with `.`
