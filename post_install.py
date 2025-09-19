#!/usr/bin/env python3
"""
Post-install helper for TELF.

- Installs (or verifies) spaCy + models and NLTK data in the *current* Python env.
- Optional GPU/HPC helpers via flags (uses conda consistently with -y).
- Avoids changing NumPy versions or bypassing your resolver.

Usage examples:
  python post_install.py
  python post_install.py --gpu
  python post_install.py --hpc-conda
  python post_install.py --gpu --gpu-toolkit
"""

import argparse
import importlib.util
import subprocess
import sys
from shutil import which


def run(cmd, **kw):
    """Run a command with check=True and echo it."""
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def ensure_pkg(import_name: str, pip_name: str | None = None, version: str | None = None):
    """
    Ensure `import_name` can be imported. If not, pip-install into THIS interpreter.
    """
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        to_install = pip_name or import_name
        if version:
            to_install = f"{to_install}=={version}"
        run([sys.executable, "-m", "pip", "install", to_install])


def ensure_spacy_and_models():
    """
    Make sure spaCy is present and large/transformer models are available.
    Model downloads are skipped if already installed.
    """
    # Keep versions aligned with your pyproject (adjust if you bump there)
    ensure_pkg("spacy", "spacy", "3.8.2")

    # Only install nltk if we’re actually going to download NLTK data later.
    # We do it here so the import is guaranteed to work for the downloader.
    ensure_pkg("nltk", "nltk", "3.9.1")

    # Install spaCy models only if missing
    if not has_module("en_core_web_lg"):
        run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    else:
        print("spaCy model en_core_web_lg already present; skipping.")

    if not has_module("en_core_web_trf"):
        run([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
    else:
        print("spaCy model en_core_web_trf already present; skipping.")


def download_nltk_data():
    """
    Download NLTK corpora via API (more robust than `-m nltk.downloader`).
    Skips re-downloads.
    """
    import nltk

    for pkg in ("wordnet", "omw-1.4"):
        print(f"Ensuring NLTK data: {pkg}")
        nltk.download(pkg, quiet=True)


def conda_required():
    if which("conda") is None:
        raise RuntimeError(
            "Conda is required for the requested GPU/HPC (conda) operations, "
            "but 'conda' was not found on PATH."
        )


def conda_install(*packages: str, channel: str = "conda-forge"):
    """
    Install conda packages non-interactively from a consistent channel.
    """
    conda_required()
    run(["conda", "install", "-y", "-c", channel, *packages])


def install_gpu_dependencies(via_conda_toolkit: bool, install_cupy: bool):
    """
    Optionally install CUDA toolkit pieces and CuPy. Uses conda-forge consistently.
    """
    if via_conda_toolkit:
        print("Installing CUDA toolkit components (conda-forge)...")
        conda_install("cudatoolkit", channel="conda-forge")
        conda_install("cudnn", channel="conda-forge")

    if install_cupy:
        print("Installing CuPy (conda-forge)...")
        conda_install("cupy", channel="conda-forge")


def install_mpi(hpc_pip: bool, hpc_conda: bool):
    """
    Optionally install mpi4py either via pip (requires system MPI toolchain)
    or via conda-forge (preferred for portability).
    """
    if hpc_pip and hpc_conda:
        # Prefer conda-forge to avoid system MPI mismatches
        print("Both --hpc and --hpc-conda were set; preferring conda-forge build.")
        hpc_pip = False

    if hpc_conda:
        print("Installing mpi4py via conda-forge...")
        conda_install("mpi4py", channel="conda-forge")
    elif hpc_pip:
        print("Installing mpi4py via pip (requires system MPI headers/libs)...")
        run([sys.executable, "-m", "pip", "install", "mpi4py"])


def run_post_install_commands(
    gpu: bool = False,
    hpc: bool = False,
    hpc_conda: bool = False,
    gpu_toolkit: bool = False,
    skip_models: bool = False,
):
    """
    Execute post-install steps in a safe, idempotent way.
    """
    # 1) NLP bits (spaCy + models, NLTK data)
    if not skip_models:
        ensure_spacy_and_models()
        download_nltk_data()
    else:
        print("Skipping spaCy model and NLTK data steps (--skip-models).")

    # 2) GPU deps (optional)
    if gpu_toolkit or gpu:
        install_gpu_dependencies(via_conda_toolkit=gpu_toolkit, install_cupy=gpu)

    # 3) HPC MPI (optional)
    install_mpi(hpc_pip=hpc, hpc_conda=hpc_conda)

    print("Post-install completed successfully.")


def main():
    p = argparse.ArgumentParser(
        description="Post installation script for downloading models/data and optional GPU/HPC extras."
    )
    p.add_argument("--gpu", action="store_true", help="Install CuPy via conda-forge.")
    p.add_argument(
        "--gpu-toolkit",
        action="store_true",
        help="Install cudatoolkit and cudnn via conda-forge.",
    )
    p.add_argument(
        "--hpc",
        action="store_true",
        help="Install mpi4py via pip (requires compatible system MPI).",
    )
    p.add_argument(
        "--hpc-conda",
        action="store_true",
        help="Install mpi4py via conda-forge (preferred for portability).",
    )
    p.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip spaCy model downloads and NLTK data steps.",
    )

    args = p.parse_args()
    try:
        run_post_install_commands(
            gpu=args.gpu,
            hpc=args.hpc,
            hpc_conda=args.hpc_conda,
            gpu_toolkit=args.gpu_toolkit,
            skip_models=args.skip_models,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}:\n  {' '.join(e.cmd)}")
        sys.exit(e.returncode)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
