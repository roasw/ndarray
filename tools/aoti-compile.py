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
from typing import Any

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile exported algorithm modules to .pt2 packages"
    )
    parser.add_argument("--algorithm-module", required=True)
    parser.add_argument("--algorithm-class")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional export config as key=value (repeatable)",
    )
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump exported (uncompiled) graph as plain text",
    )
    parser.add_argument(
        "--mode",
        default="debug",
        help="Packaging mode: debug or release (default: debug)",
    )
    return parser.parse_args()


def parse_config(items: list[str]) -> dict[str, Any]:
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


def load_export(
    cls_module: str, cls_name: str | None, config: dict[str, Any]
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    module = importlib.import_module(cls_module)
    cls: type[nn.Module] | None = None
    if cls_name:
        candidate = getattr(module, cls_name, None)
        if candidate is None:
            raise RuntimeError(
                f"Class '{cls_name}' was not found in module '{cls_module}'"
            )
        if not inspect.isclass(candidate):
            raise RuntimeError(f"'{cls_name}' in module '{cls_module}' is not a class")
        if not issubclass(candidate, nn.Module):
            raise RuntimeError(
                f"Class '{cls_name}' in module '{cls_module}' "
                "must inherit torch.nn.Module"
            )
        if candidate.__module__ != module.__name__:
            raise RuntimeError(
                f"Class '{cls_name}' must be defined in '{cls_module}', "
                "not imported from another module"
            )
        if "export" not in candidate.__dict__:
            raise RuntimeError(
                f"Class '{cls_name}' must define export(**config) directly"
            )
        cls = candidate
    else:
        candidates: list[type[nn.Module]] = []
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != module.__name__:
                continue
            if not issubclass(candidate, nn.Module):
                continue
            if "export" not in candidate.__dict__:
                continue
            export_attr = candidate.__dict__["export"]
            if isinstance(export_attr, (classmethod, staticmethod)):
                export_callable = export_attr.__func__
            else:
                export_callable = export_attr
            if not callable(export_callable):
                continue
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError(
                "Could not auto-detect algorithm class in module "
                f"'{cls_module}'. Expected exactly one class defined in this "
                "module that inherits torch.nn.Module and defines "
                "export(**config)."
            )
        if len(candidates) > 1:
            candidate_names = ", ".join(sorted(c.__name__ for c in candidates))
            raise RuntimeError(
                "Found multiple algorithm class candidates in module "
                f"'{cls_module}': {candidate_names}. "
                "Pass --algorithm-class explicitly."
            )

        cls = candidates[0]

    assert cls is not None
    if not hasattr(cls, "export"):
        raise RuntimeError(f"{cls.__name__} does not define export(**config)")
    export_payload = cls.export(**config)
    if not isinstance(export_payload, dict):
        raise RuntimeError("export() must return a dict")
    return export_payload


def validate_export_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise RuntimeError("export payload cannot be empty")
    if not all(isinstance(name, str) for name in payload):
        raise RuntimeError("export payload keys must be model names (str)")
    return payload


def canonical_variant_name(model_name: str, algorithm_name: str) -> str:
    prefix = f"{algorithm_name}_"
    if not model_name.startswith(prefix):
        raise RuntimeError(
            "Exported model name must start with algorithm basename prefix "
            f"'{prefix}', got '{model_name}'"
        )
    suffix = model_name[len(prefix) :]

    known_variants = {
        "cpu_f32_model": "cpu_f32",
        "cpu_f64_model": "cpu_f64",
        "cuda_f32_model": "cuda_f32",
        "cuda_f64_model": "cuda_f64",
    }
    if suffix not in known_variants:
        raise RuntimeError(
            f"Unsupported exported model suffix '{suffix}' for '{model_name}'"
        )
    return known_variants[suffix]


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"debug", "release"}:
        raise RuntimeError("--mode must be either 'debug' or 'release'")
    return normalized


def dump_exported_graph(name: str, module: Any, output_dir: Path) -> None:
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
    with zipfile.ZipFile(package_path, "r") as zf:
        cpp_entries = [name for name in zf.namelist() if name.endswith(".cpp")]

        if mode == "debug":
            if not cpp_entries:
                raise RuntimeError(
                    f"Debug mode expected .cpp sources in {package_path}"
                )
            if all(
                b"source stripped in release mode" in zf.read(name)
                for name in cpp_entries
            ):
                raise RuntimeError(
                    f"Debug mode found stripped .cpp sources in {package_path}"
                )
            return

        # release mode
        for name in cpp_entries:
            data = zf.read(name)
            if b"source stripped in release mode" not in data:
                raise RuntimeError(f"Release mode found unstripped source file: {name}")


def main() -> int:
    args = parse_args()
    mode = normalize_mode(args.mode)
    config = parse_config(args.config)
    payload = load_export(args.algorithm_module, args.algorithm_class, config)
    modules = validate_export_payload(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    algorithm_name = Path(args.algorithm_module.split(".")[-1]).stem
    metadata_stem = args.metadata_path.stem
    expected_metadata_stem = algorithm_name
    if metadata_stem != expected_metadata_stem:
        raise RuntimeError(
            "Metadata basename must match algorithm basename convention: "
            f"expected '{expected_metadata_stem}', got '{metadata_stem}'"
        )

    package_map: dict[str, str] = {}
    for name in sorted(modules.keys()):
        if args.dump:
            dump_exported_graph(name, modules[name], args.output_dir)
        package_path = args.output_dir / f"{name}.pt2"
        torch._inductor.aoti_compile_and_package(
            modules[name], package_path=str(package_path)
        )
        if mode == "release":
            strip_cpp_sources_in_package(package_path)
        verify_package_mode(package_path, mode)
        variant_name = canonical_variant_name(name, algorithm_name)
        package_map[variant_name] = str(package_path)

    metadata_lines = [f"mode={mode}"]
    for variant_name in sorted(package_map.keys()):
        metadata_lines.append(f"{variant_name}={package_map[variant_name]}")

    args.metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
