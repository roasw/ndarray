#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile exported algorithm modules to .pt2 packages"
    )
    parser.add_argument("--algorithm-module", required=True)
    parser.add_argument("--algorithm-class", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    cls_module: str, cls_name: str, config: dict[str, Any]
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    module = importlib.import_module(cls_module)
    cls = getattr(module, cls_name)
    if not hasattr(cls, "export"):
        raise RuntimeError(f"{cls_name} does not define export(**config)")
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


def main() -> int:
    args = parse_args()
    config = parse_config(args.config)
    payload = load_export(args.algorithm_module, args.algorithm_class, config)
    modules = validate_export_payload(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(modules.keys()):
        if args.dump:
            dump_exported_graph(name, modules[name], args.output_dir)
        package_path = args.output_dir / f"{name}.pt2"
        torch._inductor.aoti_compile_and_package(
            modules[name], package_path=str(package_path)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
