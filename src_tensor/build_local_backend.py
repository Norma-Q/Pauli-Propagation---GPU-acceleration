from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_tensor_dir = repo_root / "src_tensor"
    source = src_tensor_dir / "pps_tensor_backend_local_build.cpp"

    setup(
        name="pps_tensor_backend_local_build",
        ext_modules=[
            CppExtension(
                name="src_tensor._pps_tensor_backend_local",
                sources=[str(source)],
                extra_compile_args={"cxx": ["-O3"]},
            )
        ],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
        script_args=["build_ext", "--inplace"],
    )

    matches = sorted(src_tensor_dir.glob("_pps_tensor_backend_local*.so"))
    if not matches:
        raise RuntimeError("Build finished but no _pps_tensor_backend_local*.so was produced")

    built_path = next((path for path in matches if path.name != "_pps_tensor_backend_local.so"), matches[0])
    legacy_path = src_tensor_dir / "_pps_tensor_backend_local.so"
    if built_path != legacy_path:
        if legacy_path.exists():
            legacy_path.unlink()
        shutil.copy2(built_path, legacy_path)
    for stale in matches:
        if stale != legacy_path:
            stale.unlink()

    so_path = legacy_path
    spec = importlib.util.spec_from_file_location("src_tensor._pps_tensor_backend_local", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {so_path}")
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)

    required = [
        "build_clifford_step_cpp",
        "build_clifford_step_mw_cpp",
        "build_pauli_rotation_step_cpp",
        "build_pauli_rotation_step_mw_cpp",
        "build_pauli_rotation_step_implicit_cpp",
        "build_pauli_rotation_step_implicit_mw_cpp",
        "build_depolarizing_step_cpp",
        "build_depolarizing_step_mw_cpp",
        "build_amplitude_damping_step_cpp",
        "build_amplitude_damping_step_mw_cpp",
    ]
    missing = [name for name in required if not hasattr(imported, name)]
    if missing:
        raise RuntimeError(f"Built local backend missing expected symbols: {missing}")

    print(f"[Local Backend] Built backend: {so_path}")


if __name__ == "__main__":
    main()
