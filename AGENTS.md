# AGENTS.md

## Scope

- `ndarray`: C++ column-major DLPack container with Python interop.
- Upsampling pipeline: Python export -> AOTI `.pt2` -> C++ runtime.
- New work: C++ torch CPU kernels under `src/kernel`, registered to torch and usable by AOT export/compile.

## Mandatory workflow

- Always ask for confirmation before applying code changes.
- Build/test after edits:
  - `cmake --build build/Debug`
  - `ctest --test-dir build/Debug --output-on-failure`
- Run linters before commit: `pre-commit run --all`.
- Keep zero-copy behavior as a hard requirement for ndarray/torch DLPack interop.

## Project layout

- Headers: `inc/container`, `inc/algorithm`.
- Sources: `src/container`, `src/algorithm`, `src/kernel`.
- Python: package `python/ndarry`.
- Tests: `tests/demo`, `tests/ndarray`, `tests/algorithm`, `tests/python`.

## Documentation

- Doxygen docs are generated from `inc/container/ndarray.hpp`.
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
