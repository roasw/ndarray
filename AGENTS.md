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
- Python: `python/algorithm`, `python/tests`.
- Tests: `tests/demo`, `tests/ndarray`, `tests/algorithm`, `tests/kernel`.

## Kernel constraints (torch CPU kernel)

- Register custom ops with torch dispatcher (`TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`).
- Keep kernel CPU-only for now; reserve naming for future CUDA variants.
- Ensure Python export path can call the registered op (so `torch.export` and AOTI can compile it).
- Add parity tests against current algorithm behavior and C++ runtime coverage.
- Do not introduce copies across ndarray\<->torch bridges unless explicitly required.

## Style & conventions

- C++: existing formatting patterns, `#pragma once`, methods `UpperCamelCase`.
- Private class members use `m_` prefix; struct/public fields use `lowerCamelCase`.
- Header include order: STL, blank, third-party, blank, project.
- Keep public headers declaration-focused and readable.
- `ndarray` remains templated with explicit instantiation in `.cpp`.

## Commits

- Message format: `(chore|doc|fix|feat|infra|refac|revert): <description>.`
- Max 72 chars, must end with `.`
