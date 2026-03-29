# AGENTS.md

## Project Overview

C++ project for creating column-major DLPack arrays using LibTorch's c10 allocator, with Armadillo support.

## Build System

Uses CMakePresets.json with Ninja generator. Configure and build:
- Debug: `cmake --preset Debug && cmake --build build/Debug`
- Release: `cmake --preset Release && cmake --build build/Release`

## Code Style

- C++ formatting: Follow existing patterns in codebase
- Run `ruff` for any Python files

## Pre-commit Hooks

Linters run automatically via git-hooks.nix on commit:
- `nixfmt`, `gitlint`, `clang-format`, `ruff`, `ruff-format`

## Commit Messages

Format: `(chore|doc|fix|feat|infra|refac|revert): <description>.`
- Max title length: 72 characters
- Must end with a period

## Dependencies

Provided by nix devShell: libtorch-bin, armadillo, cmake, ninja

## Technical Notes

### DLPack with Column-Major Layout

When creating DLPack arrays for Armadillo interoperability:
- Use c10 allocator from LibTorch for memory management
- Set strides for column-major (Fortran) order
- Ensure proper memory alignment for tensor operations

## Testing

- Test folder: `test/`
- Tests are built and run alongside the main project

## Commands to Run After Changes

1. Build: `cmake --build build/Debug`
2. Run tests: `ctest --test-dir build/Debug`
