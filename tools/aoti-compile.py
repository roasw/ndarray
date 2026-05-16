#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import torch
import torch.nn as nn


KNOWN_VARIANTS = {
    "cpu_f32_model": "cpu_f32",
    "cpu_f64_model": "cpu_f64",
    "cuda_f32_model": "cuda_f32",
    "cuda_f64_model": "cuda_f64",
}

STRIPPED_SOURCE_SENTINEL = b"source stripped in release mode"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AOTI export packaging."""
    parser = argparse.ArgumentParser(
        description="Compile exported algorithm modules to .pt2 packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Notes:
              - If --algorithm-class is omitted, the script auto-detects exactly one
                class defined in the module that inherits torch.nn.Module and defines
                export(**config).
              - Exported model names must be prefixed with <algorithm_name>_ where
                algorithm_name is the module basename.

            Examples:
              aoti-compile.py \
                --algorithm-module python.ndarry.algorithm.upsample_2d_fourier \
                --output-dir build/Debug/artifacts \
                --metadata-path build/Debug/artifacts/upsample_2d_fourier.txt \
                --mode Debug \
                --config max_factor=8

              aoti-compile.py \
                --algorithm-module python.ndarry.algorithm.upsample_2d_fourier_kernel \
                --output-dir build/Debug/artifacts \
                --metadata-path build/Debug/artifacts/upsample_2d_fourier_kernel.txt \
                --mode Release \
                --config max_factor=8 \
                --config kernel_lib=/path/to/libndarray_torch_kernels.so
            """
        ),
    )
    parser.add_argument(
        "--algorithm-module",
        required=True,
        help="Python module path containing the algorithm export class",
    )
    parser.add_argument(
        "--algorithm-class",
        help=(
            "Optional explicit class name. If omitted, auto-detect a unique "
            "nn.Module subclass defined in the module with export(**config)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where .pt2 packages and optional graph dumps are written",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Output metadata file path (basename must match algorithm name)",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional export config entry as key=value (repeatable)",
    )
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write exported graph/code text to <model>.exported.txt",
    )
    parser.add_argument(
        "--mode",
        default="Debug",
        metavar="{Debug,Release}",
        help="Packaging mode: Debug or Release (default: Debug)",
    )
    return parser.parse_args()


def parse_config(items: list[str]) -> dict[str, Any]:
    """Parse repeated KEY=VALUE config arguments into a dictionary."""
    config: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise RuntimeError(f"Invalid config '{item}', expected key=value")
        key, value_text = item.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeError("Config key cannot be empty")
        value_text = value_text.strip()
        try:
            value: Any = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            value = value_text
        config[key] = value
    return config


def ensure_project_root_on_syspath() -> None:
    """Ensure project root is importable for dynamic module loading."""
    project_root = Path(__file__).resolve().parents[1]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _has_direct_callable_export(cls: type[nn.Module]) -> bool:
    """Return True when class defines callable export directly in class dict."""
    export_attr = cls.__dict__.get("export")
    if export_attr is None:
        return False
    if isinstance(export_attr, (classmethod, staticmethod)):
        return callable(export_attr.__func__)
    return callable(export_attr)


def _collect_algorithm_candidates(module: Any) -> list[type[nn.Module]]:
    """Collect algorithm class candidates defined in a module."""
    candidates: list[type[nn.Module]] = []
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ != module.__name__:
            continue
        if not issubclass(candidate, nn.Module):
            continue
        if not _has_direct_callable_export(candidate):
            continue
        candidates.append(candidate)
    return candidates


def _resolve_algorithm_class(module: Any, cls_name: str | None) -> type[nn.Module]:
    """Resolve algorithm class from explicit name or module auto-detection."""
    module_name = str(module.__name__)
    if cls_name:
        candidate = getattr(module, cls_name, None)
        if candidate is None:
            raise RuntimeError(
                f"Class '{cls_name}' was not found in module '{module_name}'"
            )
        if not inspect.isclass(candidate):
            raise RuntimeError(f"'{cls_name}' in module '{module_name}' is not a class")
        if not issubclass(candidate, nn.Module):
            raise RuntimeError(
                f"Class '{cls_name}' in module '{module_name}' "
                "must inherit torch.nn.Module"
            )
        if candidate.__module__ != module.__name__:
            raise RuntimeError(
                f"Class '{cls_name}' must be defined in '{module_name}', "
                "not imported from another module"
            )
        if not _has_direct_callable_export(candidate):
            raise RuntimeError(
                f"Class '{cls_name}' must define callable export(**config) directly"
            )
        return candidate

    candidates = _collect_algorithm_candidates(module)
    if not candidates:
        raise RuntimeError(
            "Could not auto-detect algorithm class in module "
            f"'{module_name}'. Expected exactly one class defined in this "
            "module that inherits torch.nn.Module and defines "
            "export(**config)."
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(sorted(c.__name__ for c in candidates))
        raise RuntimeError(
            "Found multiple algorithm class candidates in module "
            f"'{module_name}': {candidate_names}. "
            "Pass --algorithm-class explicitly."
        )
    return candidates[0]


def load_export(
    cls_module: str, cls_name: str | None, config: dict[str, Any]
) -> dict[str, Any]:
    """Load algorithm class and call export(**config)."""
    ensure_project_root_on_syspath()

    module = importlib.import_module(cls_module)
    cls = _resolve_algorithm_class(module, cls_name)
    if not hasattr(cls, "export"):
        raise RuntimeError(f"{cls.__name__} does not define export(**config)")
    export_payload = cls.export(**config)
    if not isinstance(export_payload, dict):
        raise RuntimeError("export() must return a dict")
    return export_payload


def validate_export_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate basic structure of export payload."""
    if not payload:
        raise RuntimeError("export payload cannot be empty")
    if not all(isinstance(name, str) for name in payload):
        raise RuntimeError("export payload keys must be model names (str)")
    return payload


def canonical_variant_name(model_name: str, algorithm_name: str) -> str:
    """Convert exported model name suffix into metadata variant key."""
    prefix = f"{algorithm_name}_"
    if not model_name.startswith(prefix):
        raise RuntimeError(
            "Exported model name must start with algorithm basename prefix "
            f"'{prefix}', got '{model_name}'"
        )
    suffix = model_name[len(prefix) :]

    if suffix not in KNOWN_VARIANTS:
        raise RuntimeError(
            f"Unsupported exported model suffix '{suffix}' for '{model_name}'"
        )
    return KNOWN_VARIANTS[suffix]


def normalize_mode(mode: str) -> str:
    """Normalize and validate packaging mode (accepts Debug/debug/DEBUG etc.)."""
    normalized = mode.strip().lower()
    if normalized not in {"debug", "release"}:
        raise RuntimeError("--mode must be 'Debug' or 'Release'")
    return normalized


def dump_exported_graph(name: str, module: Any, output_dir: Path) -> None:
    """Write exported graph signature, graph, and code to a text dump."""
    dump_path = output_dir / f"{name}.exported.txt"
    graph_module = getattr(module, "graph_module", None)
    if graph_module is None:
        raise RuntimeError(f"Exported module '{name}' has no graph_module")

    graph_text = str(graph_module.graph)
    code_text = str(graph_module.code)
    signature_text = str(getattr(module, "graph_signature", ""))

    dump_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "",
                "[graph_signature]",
                signature_text,
                "",
                "[graph]",
                graph_text,
                "",
                "[code]",
                code_text,
                "",
            ]
        ),
        encoding="utf-8",
    )


def strip_cpp_sources_in_package(package_path: Path) -> None:
    """Replace embedded .cpp sources in a .pt2 package with a sentinel."""
    tmp_dir = tempfile.mkdtemp(dir=str(package_path.parent))
    try:
        rewritten = Path(tmp_dir) / "rewritten.pt2"
        with (
            zipfile.ZipFile(package_path, "r") as zin,
            zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_STORED) as zout,
        ):
            for name in zin.namelist():
                if name.endswith(".cpp"):
                    zout.writestr(name, "// source stripped in release mode\n")
                    continue
                zout.writestr(name, zin.read(name))
        shutil.move(str(rewritten), str(package_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_package_mode(package_path: Path, mode: str) -> None:
    """Verify debug/release package source stripping expectations."""
    with zipfile.ZipFile(package_path, "r") as zf:
        cpp_entries = [name for name in zf.namelist() if name.endswith(".cpp")]

        if mode == "debug":
            if not cpp_entries:
                raise RuntimeError(
                    f"Debug mode expected .cpp sources in {package_path}"
                )
            if all(STRIPPED_SOURCE_SENTINEL in zf.read(name) for name in cpp_entries):
                raise RuntimeError(
                    f"Debug mode found stripped .cpp sources in {package_path}"
                )
            return

        # release mode
        for name in cpp_entries:
            data = zf.read(name)
            if STRIPPED_SOURCE_SENTINEL not in data:
                raise RuntimeError(f"Release mode found unstripped source file: {name}")


def algorithm_name_from_module(module_name: str) -> str:
    """Return algorithm basename from fully-qualified module name."""
    return Path(module_name.split(".")[-1]).stem


def validate_metadata_stem(metadata_path: Path, algorithm_name: str) -> None:
    """Ensure metadata basename matches algorithm naming contract."""
    metadata_stem = metadata_path.stem
    if metadata_stem != algorithm_name:
        raise RuntimeError(
            "Metadata basename must match algorithm basename convention: "
            f"expected '{algorithm_name}', got '{metadata_stem}'"
        )


def compile_and_collect_packages(
    modules: dict[str, Any],
    output_dir: Path,
    mode: str,
    algorithm_name: str,
    dump: bool,
) -> dict[str, str]:
    """Compile exported modules to .pt2 and collect metadata variant mapping."""
    package_map: dict[str, str] = {}
    for model_name in sorted(modules.keys()):
        exported_module = modules[model_name]
        if dump:
            dump_exported_graph(model_name, exported_module, output_dir)

        package_path = output_dir / f"{model_name}.pt2"

        inductor_configs: dict[str, Any] = {}
        if mode == "release":
            inductor_configs["debug"] = False

        torch._inductor.aoti_compile_and_package(
            exported_module,
            package_path=str(package_path),
            inductor_configs=inductor_configs if inductor_configs else None,
        )

        if mode == "release":
            strip_cpp_sources_in_package(package_path)
        verify_package_mode(package_path, mode)

        variant_name = canonical_variant_name(model_name, algorithm_name)
        package_map[variant_name] = str(package_path)

    return package_map


def write_metadata_file(
    metadata_path: Path, mode: str, package_map: dict[str, str]
) -> None:
    """Write package metadata manifest file."""
    metadata_lines = [f"mode={mode}"]
    for variant_name in sorted(package_map.keys()):
        metadata_lines.append(f"{variant_name}={package_map[variant_name]}")
    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")


def main() -> int:
    """Compile exported modules and write metadata manifest."""
    args = parse_args()
    mode = normalize_mode(args.mode)
    config = parse_config(args.config)
    payload = load_export(args.algorithm_module, args.algorithm_class, config)
    modules = validate_export_payload(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    algorithm_name = algorithm_name_from_module(args.algorithm_module)
    validate_metadata_stem(args.metadata_path, algorithm_name)

    package_map = compile_and_collect_packages(
        modules=modules,
        output_dir=args.output_dir,
        mode=mode,
        algorithm_name=algorithm_name,
        dump=args.dump,
    )
    write_metadata_file(args.metadata_path, mode, package_map)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
