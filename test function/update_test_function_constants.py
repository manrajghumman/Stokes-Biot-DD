#!/usr/bin/env python3
"""
Update manufactured-solution Function constants in biot.prm and stokes.prm
from parameters_sb.prm.

Usage:
  python3 update_test_function_constants.py parameters_sb.prm biot.prm stokes.prm

Biot mapping:
  Biot Parameters / viscosity   -> mu
  Biot Parameters / s_0         -> s0
  Biot Parameters / alpha       -> alpha
  Biot Parameters / alpha_BJS   -> alpha_BJS
  Biot Parameters / lame_mu     -> mu_p
  Biot Parameters / lame_lambda -> lambda_p
  Biot Parameters / K           -> k
    Only isotropic scalar/diagonal K is supported for the manufactured file.

Stokes mapping:
  Stokes Parameters / viscosity -> mu
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


def read_subsection_parameters(parameters_file: Path, subsection_name: str) -> dict[str, str]:
    params: dict[str, str] = {}
    in_subsection = False

    for raw_line in parameters_file.read_text().splitlines():
        line = raw_line.strip()

        if re.match(rf"^subsection\s+{re.escape(subsection_name)}\s*$", line):
            in_subsection = True
            continue

        if in_subsection and line == "end":
            break

        if not in_subsection:
            continue

        parsed = parse_assignment_value(raw_line)
        if parsed is not None:
            name, value = parsed
            params[name] = value

    if not in_subsection:
        raise ValueError(f"Could not find subsection `{subsection_name}` in {parameters_file}")

    return params


def scalar_k_from_K(K_value: str) -> str:
    """
    Extract scalar k from an isotropic K value.

    Accepts either:
      1e-4
    or
      1e-4, 0.0; 0.0, 1e-4
    """
    rows = [[entry.strip() for entry in row.split(",")] for row in K_value.split(";")]

    if len(rows) == 1 and len(rows[0]) == 1:
        return rows[0][0]

    if len(rows) < 2 or len(rows[0]) < 2 or len(rows[1]) < 2:
        raise ValueError(f"Could not parse K as scalar or 2x2 matrix: {K_value!r}")

    k11 = rows[0][0]
    k22 = rows[1][1]

    try:
        if float(k11) != float(k22):
            raise ValueError(
                "The manufactured-solution file uses scalar `k`, but K is not isotropic: "
                f"K_11={k11}, K_22={k22}"
            )
    except ValueError:
        # If conversion failed but strings are exactly equal, still accept.
        if k11 != k22:
            raise

    return k11


def make_biot_constants(params: dict[str, str]) -> str:
    required = ["viscosity", "s_0", "alpha", "alpha_BJS", "lame_mu", "lame_lambda", "K"]
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"Missing required Biot parameter(s): {', '.join(missing)}")

    k = scalar_k_from_K(params["K"])
    return (
        f"mu={params['viscosity']}, k={k}, s0={params['s_0']}, alpha={params['alpha']}, "
        f"alpha_BJS={params['alpha_BJS']}, mu_p={params['lame_mu']}, "
        f"lambda_p={params['lame_lambda']}"
    )


def make_stokes_constants(params: dict[str, str]) -> str:
    if "viscosity" not in params:
        raise ValueError("Missing required Stokes parameter: viscosity")
    return f"mu={params['viscosity']}"


def update_all_existing_function_constants(file_path: Path, constants: str) -> int:
    """Replace every existing `set Function constants = ...` line in a file."""
    text = file_path.read_text()

    pattern = re.compile(r"^(\s*set\s+Function constants\s*=\s*).*$", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{constants}", text)

    if count == 0:
        raise ValueError(f"No `set Function constants = ...` lines found in {file_path}")

    if new_text != text:
        file_path.write_text(new_text)

    return count


def ensure_function_constants_in_subsection(text: str, subsection_name: str, constants: str) -> tuple[str, int]:
    """
    In a named subsection, replace `set Function constants = ...` if present;
    otherwise insert one immediately after the subsection line.
    """
    block_pattern = re.compile(
        rf"(?ms)^(\s*subsection\s+{re.escape(subsection_name)}\s*\n)(.*?)(^\s*end\s*$)"
    )

    constants_pattern = re.compile(
        r"^(\s*set\s+Function constants\s*=\s*).*$",
        re.MULTILINE,
    )

    changed_blocks = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed_blocks

        header, body, end_line = match.group(1), match.group(2), match.group(3)

        if constants_pattern.search(body):
            body_new, n = constants_pattern.subn(rf"\g<1>{constants}", body)
            changed_blocks += n
        else:
            body_new = f"    set Function constants = {constants}\n" + body
            changed_blocks += 1

        return header + body_new + end_line

    new_text, n_blocks = block_pattern.subn(repl, text, count=1)

    if n_blocks == 0:
        raise ValueError(f"Could not find subsection `{subsection_name}`")

    return new_text, changed_blocks


def update_stokes_file(stokes_file: Path, constants: str) -> int:
    """
    Update/insert Stokes `mu` constants where the manufactured Stokes file uses it.

    RHS vector should use mu in the body force.
    BC Neumann should use mu in the stress tensor if Neumann boundaries are tested.
    """
    text = stokes_file.read_text()
    total_count = 0

    for subsection in ["RHS vector", "BC Neumann"]:
        text, count = ensure_function_constants_in_subsection(text, subsection, constants)
        total_count += count

    if total_count == 0:
        raise ValueError(f"No Stokes constants updated in {stokes_file}")

    if text != stokes_file.read_text():
        stokes_file.write_text(text)

    return total_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters_file", type=Path)
    parser.add_argument("biot_file", type=Path)
    parser.add_argument("stokes_file", type=Path)

    args = parser.parse_args()

    biot_params = read_subsection_parameters(args.parameters_file, "Biot Parameters")
    stokes_params = read_subsection_parameters(args.parameters_file, "Stokes Parameters")

    biot_constants = make_biot_constants(biot_params)
    stokes_constants = make_stokes_constants(stokes_params)

    n_biot = update_all_existing_function_constants(args.biot_file, biot_constants)
    n_stokes = update_stokes_file(args.stokes_file, stokes_constants)

    print(f"Updated {n_biot} Function constants line(s) in {args.biot_file}")
    print(f"Biot Function constants = {biot_constants}")
    print(f"Updated {n_stokes} Function constants line(s) in {args.stokes_file}")
    print(f"Stokes Function constants = {stokes_constants}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"update_test_function_constants.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
