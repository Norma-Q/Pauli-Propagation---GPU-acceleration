from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.cpp_extension import load


def _tensor_equal(a, b, atol: float = 1e-10) -> tuple[bool, str]:
    if a is None and b is None:
        return True, "both None"
    if (a is None) != (b is None):
        return False, "None mismatch"
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return (a == b), f"non-tensor mismatch: {a} vs {b}"
    if a.shape != b.shape:
        return False, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return False, f"dtype mismatch {a.dtype} vs {b.dtype}"
    if a.numel() == 0 and b.numel() == 0:
        return True, "empty"
    if a.dtype.is_floating_point:
        ok = bool(torch.allclose(a, b, atol=atol, rtol=0.0))
    else:
        ok = bool(torch.equal(a, b))
    return ok, ("ok" if ok else "value mismatch")


def _tuple_equal(lhs, rhs, name: str) -> tuple[bool, str]:
    if len(lhs) != len(rhs):
        return False, f"{name}: tuple size mismatch {len(lhs)} vs {len(rhs)}"
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        ok, msg = _tensor_equal(a, b)
        if not ok:
            return False, f"{name}[{i}] {msg}"
    return True, f"{name}: equal"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src_tensor" / "pps_tensor_backend.cpp"
    patched = repo / "src_tensor" / "pps_tensor_backend_candidate.cpp"
    build_dir = repo / ".build" / "pps_backend_compare"
    build_dir.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8")
    prelude = "#include <time.h>\nextern \"C\" int timespec_get(struct timespec*, int);\n"
    text = prelude + text
    text = text.replace(
        "PYBIND11_MODULE(_pps_tensor_backend, m)",
        "PYBIND11_MODULE(_pps_tensor_backend_candidate, m)",
    )
    patched.write_text(text, encoding="utf-8")

    candidate = load(
        name="_pps_tensor_backend_candidate",
        sources=[str(patched)],
        build_directory=str(build_dir),
        verbose=True,
        extra_cflags=[
            "-O3",
            "-std=gnu++17",
            "-D_GLIBCXX_HAVE_TIMESPEC_GET=1",
            "-D_ISOC11_SOURCE",
        ],
    )

    local = importlib.import_module("src_tensor._pps_tensor_backend_local")

    local_attrs = {a for a in dir(local) if a.startswith("build_")}
    candidate_attrs = {a for a in dir(candidate) if a.startswith("build_")}

    print("local build_*:", sorted(local_attrs))
    print("cand  build_*:", sorted(candidate_attrs))
    print("missing in candidate:", sorted(local_attrs - candidate_attrs))
    print("extra in candidate:", sorted(candidate_attrs - local_attrs))

    torch.manual_seed(7)
    coeff_dtype = torch.float64
    try:
        coeff_dtype = torch._C._te.ScalarType.Double  # type: ignore[attr-defined]
    except Exception:
        coeff_dtype = torch.float64

    n_terms = 128
    x_mask = torch.randint(0, 2**20, (n_terms,), dtype=torch.int64)
    z_mask = torch.randint(0, 2**20, (n_terms,), dtype=torch.int64)
    coeffs = torch.randn(n_terms, dtype=torch.float64)
    thetas = torch.randn(20, dtype=torch.float64)

    n_words = 2
    x_mask_mw = torch.randint(0, 2**62, (n_terms, n_words), dtype=torch.int64)
    z_mask_mw = torch.randint(0, 2**62, (n_terms, n_words), dtype=torch.int64)
    coeffs_mw = torch.randn(n_terms, dtype=torch.float64)
    gx_words = torch.tensor([0b10101, 0b10011], dtype=torch.int64)
    gz_words = torch.tensor([0b10010, 0b00101], dtype=torch.int64)

    cases: list[tuple[str, Callable]] = []
    for symbol, qubits in (("H", [3]), ("S", [11]), ("CNOT", [5, 9])):
        cases.append(
            (
                f"build_clifford_step_cpp/no_prune/{symbol}",
                lambda mod, symbol=symbol, qubits=qubits: mod.build_clifford_step_cpp(
                    symbol,
                    qubits,
                    x_mask,
                    z_mask,
                    coeff_dtype,
                    None,
                    None,
                    10**9,
                    10**9,
                ),
            )
        )
        cases.append(
            (
                f"build_clifford_step_cpp/prune/{symbol}",
                lambda mod, symbol=symbol, qubits=qubits: mod.build_clifford_step_cpp(
                    symbol,
                    qubits,
                    x_mask,
                    z_mask,
                    coeff_dtype,
                    1e-7,
                    coeffs,
                    10**9,
                    10**9,
                ),
            )
        )

    cases.append(
        (
            "build_pauli_rotation_step_cpp/no_prune",
            lambda mod: mod.build_pauli_rotation_step_cpp(
                0b10101,
                0b10010,
                4,
                x_mask,
                z_mask,
                coeff_dtype,
                None,
                None,
                None,
                10**9,
                10**9,
            ),
        )
    )
    cases.append(
        (
            "build_pauli_rotation_step_cpp/prune",
            lambda mod: mod.build_pauli_rotation_step_cpp(
                0b10101,
                0b10010,
                4,
                x_mask,
                z_mask,
                coeff_dtype,
                1e-7,
                coeffs,
                thetas,
                10**9,
                10**9,
            ),
        )
    )

    for symbol, qubits in (("H", [62]), ("S", [64]), ("CNOT", [62, 64])):
        cases.append(
            (
                f"build_clifford_step_mw_cpp/no_prune/{symbol}",
                lambda mod, symbol=symbol, qubits=qubits: mod.build_clifford_step_mw_cpp(
                    symbol,
                    qubits,
                    x_mask_mw,
                    z_mask_mw,
                    coeff_dtype,
                    None,
                    None,
                    10**9,
                    10**9,
                ),
            )
        )
        cases.append(
            (
                f"build_clifford_step_mw_cpp/prune/{symbol}",
                lambda mod, symbol=symbol, qubits=qubits: mod.build_clifford_step_mw_cpp(
                    symbol,
                    qubits,
                    x_mask_mw,
                    z_mask_mw,
                    coeff_dtype,
                    1e-7,
                    coeffs_mw,
                    10**9,
                    10**9,
                ),
            )
        )

    cases.append(
        (
            "build_pauli_rotation_step_mw_cpp/no_prune",
            lambda mod: mod.build_pauli_rotation_step_mw_cpp(
                gx_words,
                gz_words,
                4,
                x_mask_mw,
                z_mask_mw,
                coeff_dtype,
                None,
                None,
                None,
                10**9,
                10**9,
            ),
        )
    )
    cases.append(
        (
            "build_pauli_rotation_step_mw_cpp/prune",
            lambda mod: mod.build_pauli_rotation_step_mw_cpp(
                gx_words,
                gz_words,
                4,
                x_mask_mw,
                z_mask_mw,
                coeff_dtype,
                1e-7,
                coeffs_mw,
                thetas,
                10**9,
                10**9,
            ),
        )
    )

    all_ok = True
    for name, fn in cases:
        out_local = fn(local)
        out_candidate = fn(candidate)
        ok, msg = _tuple_equal(out_local, out_candidate, name)
        print(("OK   " if ok else "FAIL ") + msg)
        all_ok = all_ok and ok

    print("PARITY_RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
