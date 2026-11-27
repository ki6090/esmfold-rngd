"""Compile the RNGD bucket once (fills cache/service) and check padding against CPU FP32.

Runs three passes over the test sequences and compares each to the first:
  1. original model on CPU, no padding (reference)
  2. padded language model on CPU     (padding logic alone)
  3. padded language model on RNGD    (the service path; first call compiles)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

TESTS = [
    ("chignolin", "GYDPETGTWG"),                                          # 10 aa
    ("trp_cage", "NLYIQWLKDGGPSSGRPPPS"),                                 # 20 aa
    ("ubiquitin_1_50", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQL"),  # 50 aa (full bucket)
]
START = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - START:8.1f}s] {message}", flush=True)


def compare(result, ref):
    ca, plddt = torch.tensor(result["ca"]), torch.tensor(result["plddt"])
    return {"ca_aligned_rmsd_angstrom": E.base.compare_ca(ca, ref["ca"]),
            "plddt_mean_abs_diff": float((plddt - ref["plddt"]).abs().mean()),
            "mean_plddt": result["mean_plddt"], "seconds": result["timings"]["total"]}


def main():
    engine = E.Engine(use_npu=True)
    log(f"Loaded {engine.parameters_b:.3f}B parameters; bucket={E.BUCKET_TOKENS} tokens")
    report = {"bucket_tokens": E.BUCKET_TOKENS, "recycles": engine.recycles, "tests": {}}

    refs = {}
    for name, seq in TESTS:
        out, seconds = engine.run_model(seq)
        info, ca = E.base.summarize(out, seq, seconds)
        refs[name] = {"ca": ca.clone(), "plddt": out.plddt[0, :, 1] * 100}
        report["tests"][name] = {"length": len(seq), "cpu_reference": info}
        log(f"CPU reference {name} ({len(seq)} aa): pLDDT {info['mean_plddt']:.2f}, {seconds:.1f}s")

    engine.install_padded_lm()
    for name, seq in TESTS:
        result = engine.fold(seq)
        report["tests"][name]["cpu_padded"] = compare(result, refs[name])
        log(f"CPU padded {name}: {report['tests'][name]['cpu_padded']}")

    engine.enter_context()
    engine.prepare_npu()
    log(f"NPU placement {engine.devices}; first fold compiles the {E.BUCKET_TOKENS}-token bucket")
    for name, seq in TESTS:
        result = engine.fold(seq)
        report["tests"][name]["rngd_padded"] = compare(result, refs[name])
        report["tests"][name]["rngd_timings"] = result["timings"]
        log(f"RNGD padded {name}: {report['tests'][name]['rngd_padded']}")

    path = E.ROOT / "logs/precompile_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Report written to {path}")


if __name__ == "__main__":
    main()
