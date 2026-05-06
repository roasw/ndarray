# AGENTS.md

## Project Overview

A C++/Python project that provides:

1. `ndarray`: A C++ column-major DLPack container class compatible with Armadillo and `torch.array`.
1. Upsampling algorithm: Python definition exported via `torch.compile`, compiled to a native CPU binary (`.pt2`) using `AOTIductor`, and loaded in C++.

**Important:** Always ask for confirmation before applying changes.

## Project Structure

- **Headers**: `inc/container` (`ndarray`), `inc/algorithm` (upsampling).
- **Source**: `src/container`, `src/algorithm`.
- **Tests**: `test/`.

## Development & Workflow

### Dependencies

Managed by nix `devShell` in `flake.nix`. Assume the environment is already in the `devShell` (via `direnv` integration) and all dependencies are readily available.

### Build System

Uses `CMakePresets.json` with Ninja.

- **Debug**: `cmake --preset Debug && cmake --build build/Debug`
- **Release**: `cmake --preset Release && cmake --build build/Release`

### Testing & Post-Change Commands

Tests are built and run alongside the main project. After making changes, always verify with:

1. **Build**: `cmake --build build/Debug`
1. **Test**: `ctest --test-dir build/Debug`

## Code Style & Conventions

### Formatting & Naming

- **C++ Formatting**: Follow existing patterns.
- **Header Order**: STL, empty line, third-party, empty line, project headers.
- **Private Members**: `m_` prefix + lowerCamelCase (e.g., `m_data`, `m_shape`).
- **Methods**: UpperCamelCase (e.g., `GetShape()`, `ToArmadillo()`).
- **Python**: Run `ruff` on all files.

### Pre-commit Hooks

Run `pre-commit run --all` to execute all linters. The configured linters can be looked up in `flake.nix`.

### Commit Messages

- **Format**: `(chore|doc|fix|feat|infra|refac|revert): <description>.`
- **Rules**: Max 72 characters, must end with a period.

## Technical Notes

### `ndarray` Class

- Templated with explicit specializations in `.cpp` (`float`, `double`, `int`, `bool`).
- Column-major layout for Armadillo compatibility.
- Memory managed via DLPack with LibTorch's `c10` allocator.
- Maintain NumPy compatibility.
- Python wrapper acts as a drop-in replacement for `torch.array`.

### DLPack & Column-Major Layout

For Armadillo interoperability:

- Use LibTorch's `c10` allocator.
- Set strides for column-major (Fortran) order.
- Ensure proper memory alignment.
