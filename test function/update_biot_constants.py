#!/usr/bin/env python3
"""
Update all `set Function constants = ...` lines in biot.prm from the
Biot Parameters subsection of parameters_sb.prm.

Mapping:
  s_0         -> s0
  alpha       -> alpha
  alpha_BJS   -> alpha_BJS
  lame_mu     -> mu_p
  lame_lambda -> lambda_p
  K           -> k   (only isotropic diagonal scalar is supported)

`mu` is left fixed at 1.0 unless --mu is provided.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_assignment_value(line: str) -> tuple[str, str] | None:
    """Parse lines of the form `set name = value`. Return (name, value)."""
    m = re.match(r"^\s*set\s+([^=]+?)\s*=\s*(.*?)\s*$", line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def read_biot_parameters(parameters_file: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    in_biot = False

    for raw_line in parameters_file.read_text().splitlines():
        line = raw_line.strip()

        if re.match(r"^subsection\s+Biot Parameters\s*$", line):
            in_biot = True
            continue

        if in_biot and line == "end":
            break

        if not in_biot:
            continue

        parsed = parse_assignment_value(raw_line)
        if parsed is not None:
            name, value = parsed
            params[name] = value

    required = ["s_0", "alpha", "alpha_BJS", "lame_mu", "lame_lambda", "K"]
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(
            f"Missing required Biot parameter(s) in {parameters_file}: {', '.join(missing)}"
        )

    return params


def scalar_k_from_K(K_value: str) -> str:
    """
    Extract scalar k from a 2x2 isotropic tensor written like:
      1e-4, 0.0; 0.0, 1e-4
    Raises an error if the diagonal entries differ.
    """
    rows = [[entry.strip() for entry in row.split(",")] for row in K_value.split(";")]

    if len(rows) == 1 and len(rows[0]) == 1:
        return rows[0][0]

    if len(rows) < 2 or len(rows[0]) < 2 or len(rows[1]) < 2:
        raise ValueError(f"Could not parse K as a 2x2 matrix: {K_value!r}")

    k11 = rows[0][0]
    k22 = rows[1][1]

    try:
        if float(k11) != float(k22):
            raise ValueError(
                "The manufactured-solution file uses scalar `k`, but K is not isotropic: "
                f"K_11={k11}, K_22={k22}"
            )
    except ValueError:
        # Re-raise the anisotropic error above, but still allow non-float strings only if equal.
        if k11 != k22:
            raise

    return k11


def make_constants_string(params: dict[str, str], mu: str) -> str:
    k = scalar_k_from_K(params["K"])
    return (
        f"mu={mu}, k={k}, s0={params['s_0']}, alpha={params['alpha']}, "
        f"alpha_BJS={params['alpha_BJS']}, mu_p={params['lame_mu']}, "
        f"lambda_p={params['lame_lambda']}"
    )


def update_biot_file(biot_file: Path, constants: str) -> int:
    text = biot_file.read_text()

    pattern = re.compile(r"^(\s*set\s+Function constants\s*=\s*).*$", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{constants}", text)

    if count == 0:
        raise ValueError(f"No `set Function constants = ...` lines found in {biot_file}")

    if new_text != text:
        biot_file.write_text(new_text)

    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters_file", type=Path)
    parser.add_argument("biot_file", type=Path)
    parser.add_argument("--mu", default="1.0", help="value to use for scalar mu in biot.prm")
    args = parser.parse_args()

    params = read_biot_parameters(args.parameters_file)
    constants = make_constants_string(params, args.mu)
    count = update_biot_file(args.biot_file, constants)

    print(f"Updated {count} Function constants line(s) in {args.biot_file}")
    print(f"Function constants = {constants}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"update_biot_constants.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
