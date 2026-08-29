"""CLI: fusion plan, backend info, and experiment launchers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .backend import get_backend
from .specs.models import SPECS, ModelSpec


def fusion_plan(specs: list[ModelSpec]) -> dict:
    decoder = next((s for s in specs if s.kind == "text"), None)
    if decoder is None:
        raise ValueError("a text-kind spec is required as the native decoder")
    bridges = []
    for spec in specs:
        if spec.kind == "text":
            continue
        bridges.append(
            {
                "expert": spec.name,
                "kind": spec.kind,
                "source_hidden": spec.hidden_dim,
                "target_hidden": decoder.hidden_dim,
                "projector": f"{spec.hidden_dim} -> {decoder.hidden_dim} (procrustes-init)",
                "natural_pairing": {"vision": "image <-> caption", "audio": "speech <-> transcript"}.get(spec.kind),
            }
        )
    return {
        "version": __version__,
        "decoder": {"name": decoder.name, "hidden_dim": decoder.hidden_dim},
        "experts": len(bridges),
        "bridges": bridges,
        "loss": "L_answer + l_feat + l_logit + l_contrastive + l_replay + l_missing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexuss-Fusion plan/primitives CLI")
    parser.add_argument("--specs", nargs="*", default=None, help="subset of registered specs")
    parser.add_argument("--list", action="store_true", help="list registered specs")
    parser.add_argument("--json", action="store_true", help="emit plan as JSON")
    parser.add_argument("--backend", action="store_true", help="report the active math backend")
    parser.add_argument("--run", choices=["phase2"], default=None, help="launch an experiment")
    args = parser.parse_args()

    if args.backend:
        from .backend import eigen_available

        b = get_backend()
        print(f"active backend: {b.name}")
        print(f"eigen native available: {eigen_available()}")
        sys.exit(0)

    if args.run:
        from .run.phase2 import main as phase2_main

        sys.exit(phase2_main([f"--images-dir={args.specs[0]}" if args.specs else "--images-dir=benchmarks/vision/images"]))

    if args.list:
        for name, spec in SPECS.items():
            print(f"{name}: kind={spec.kind} hidden={spec.hidden_dim} {spec.license} {spec.source}")
        sys.exit(0)

    names = args.specs or list(SPECS)
    specs = [SPECS[n] for n in names if n in SPECS]
    missing = [n for n in names if n not in SPECS]
    if missing:
        raise KeyError(f"unregistered specs: {missing}")
    plan = fusion_plan(specs)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(f"Nexuss-Fusion v{__version__}")
        print(f"native decoder: {plan['decoder']['name']} (hidden {plan['decoder']['hidden_dim']})")
        for bridge in plan["bridges"]:
            print(f"  fuse {bridge['expert']:32s} {bridge['projector']:40s} pair={bridge['natural_pairing']}")
        print("objective:", plan["loss"])


if __name__ == "__main__":
    main()