"""
validate.py — M025 Roy 2023 HIF-EPO CKD anaemia QSP model

Entry-point validation script for the evidence repository.
Runs replicate_fig3.py and checks the artifact for PASS status.

Usage:
    python validate.py

Exit codes:
    0 — PASS or INFORMATIVE_PASS
    1 — FAIL
"""
import os
import sys
import json
import subprocess
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))


def latest_artifact(evidence_dir, pattern="M025_fig3_*.json"):
    """Return path to most recent JSON artifact, or None."""
    files = sorted(glob.glob(os.path.join(evidence_dir, pattern)))
    return files[-1] if files else None


def main():
    print("=== M025 Roy 2023 Validation ===")
    evidence_dir = os.path.join(_HERE, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    print("\nRunning replicate_fig3.py ...")
    ret = subprocess.run(
        [sys.executable, os.path.join(_HERE, "replicate_fig3.py"), "--spinup", "30000"],
        cwd=_HERE,
        capture_output=False,
    )

    print(f"\nreplicate_fig3.py exit code: {ret.returncode}")

    # Find and read the latest artifact
    art_path = latest_artifact(evidence_dir)
    if art_path is None:
        print("ERROR: No artifact found in evidence/")
        sys.exit(1)

    print(f"Latest artifact: {art_path}")
    with open(art_path) as f:
        art = json.load(f)

    status = art.get("status", "UNKNOWN")
    rmse   = art.get("overall_rmse_relative")
    print(f"\nStatus: {status}")
    print(f"Overall RMSE: {rmse*100:.1f}%" if rmse else "Overall RMSE: N/A")

    # Print per-variable summary
    for var, vinfo in art.get("comparison", {}).items():
        if var.startswith("_"):
            continue
        r = vinfo.get("rmse_relative")
        tag = "(validated)" if vinfo.get("validated") else "(informative)"
        print(f"  {var}: RMSE={r*100:.1f}% {tag}" if r else f"  {var}: N/A")

    if status in ("PASS", "INFORMATIVE_PASS"):
        print(f"\nVALIDATION {status}")
        sys.exit(0)
    else:
        print(f"\nVALIDATION FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
