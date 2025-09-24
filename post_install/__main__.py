#!/usr/bin/env python3
"""
Post-install helper for TELF.

- Installs (or verifies) spaCy + models and NLTK data in the *current* Python env.
- Optional GPU/HPC helpers via flags (uses conda consistently with -y).
- Optional Kaleido/Chrome setup for Plotly static image export.
- Avoids changing NumPy versions or bypassing your resolver unless requested.

Usage examples:
  python post_install.py                         # NLP models + NLTK (default)
  python post_install.py --kaleido-chrome        # also prepare Chrome for Kaleido
  python post_install.py --gpu                   # add CuPy via conda-forge
  python post_install.py --hpc-conda             # add mpi4py via conda-forge
  python post_install.py --skip-models           # skip spaCy/NLTK bits

As a console script (when exposed via pyproject):
  telf-post-install --kaleido-chrome
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from shutil import which

# ------------------------------
# Utilities
# ------------------------------

def run(cmd, **kw):
    """Run a command with check=True and echo it."""
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(str(x) for x in cmd)
    else:
        printable = str(cmd)
    print(">", printable)
    subprocess.run(cmd, check=True, **kw)


def has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def ensure_pkg(import_name: str, pip_name: str | None = None, version_str: str | None = None):
    """
    Ensure `import_name` can be imported. If not, pip-install into THIS interpreter.
    """
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        to_install = pip_name or import_name
        if version_str:
            to_install = f"{to_install}=={version_str}"
        run([sys.executable, "-m", "pip", "install", to_install])


# ------------------------------
# NLP bits (spaCy + NLTK)
# ------------------------------

def ensure_spacy_and_models():
    """
    Make sure spaCy is present and large/transformer models are available.
    Model downloads are skipped if already installed.
    """
    # Keep versions aligned with pyproject (adjust if you bump there)
    ensure_pkg("spacy", "spacy", "3.8.2")

    # Only install nltk if we’re actually going to download NLTK data later.
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


# ------------------------------
# GPU / HPC optional helpers
# ------------------------------

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


# ------------------------------
# Kaleido / Chrome for Plotly static export
# ------------------------------

MIN_PLOTLY = "6.1.1"   # Kaleido v1 requires Plotly >= 6.1.1
MIN_KALEIDO = "1.1.0"  # Provides get_chrome_sync()

def _ensure_packaging():
    try:
        import packaging  # noqa: F401
    except ModuleNotFoundError:
        run([sys.executable, "-m", "pip", "install", "packaging"])


def _pkg_version_ok(dist: str, min_ver: str) -> bool:
    _ensure_packaging()
    from packaging.version import Version
    try:
        return Version(version(dist)) >= Version(min_ver)
    except PackageNotFoundError:
        return False


def ensure_plotly_kaleido_versions():
    """
    Ensure plotly/kaleido are installed and new enough for Kaleido v1.
    """
    if not _pkg_version_ok("plotly", MIN_PLOTLY):
        run([sys.executable, "-m", "pip", "install", f"plotly>={MIN_PLOTLY}"])
    if not _pkg_version_ok("kaleido", MIN_KALEIDO):
        run([sys.executable, "-m", "pip", "install", f"kaleido>={MIN_KALEIDO}"])


def _try_cli_chrome_fetch():
    """
    Fallback to CLI helpers if available: plotly_get_chrome / kaleido_get_chrome.
    """
    for cli in ("plotly_get_chrome", "kaleido_get_chrome"):
        path = which(cli)
        if path:
            try:
                run([path])
                return True
            except subprocess.CalledProcessError:
                pass
    return False


def ensure_chrome_for_kaleido():
    """
    Ensure Chrome is available for Kaleido v1+.

    Behavior:
      - If BROWSER_PATH points to a real file, use it.
      - Else, try kaleido.get_chrome_sync() to download a portable Chrome.
      - Else, try CLI helpers (plotly_get_chrome / kaleido_get_chrome).
      - Raises on failure.
    """
    ensure_plotly_kaleido_versions()

    bp = os.environ.get("BROWSER_PATH")
    if bp and os.path.exists(bp):
        print(f"BROWSER_PATH already set: {bp}")
        return

    # Preferred: Python helper returns a path; we also set BROWSER_PATH.
    try:
        import kaleido
        if hasattr(kaleido, "get_chrome_sync"):
            print("Preparing Chrome for Kaleido (this may download a portable binary)...")
            path = kaleido.get_chrome_sync()
            os.environ["BROWSER_PATH"] = str(path)
            print(f"✅ Chrome ready for Kaleido at: {path}")
            return
    except ModuleNotFoundError:
        # Shouldn't happen; ensure_plotly_kaleido_versions installed it.
        run([sys.executable, "-m", "pip", "install", f"kaleido>={MIN_KALEIDO}"])
        import kaleido  # noqa: F401

    # Fallback: try CLI helpers
    if _try_cli_chrome_fetch():
        print("Chrome prepared via CLI helper.")
        return

    raise RuntimeError(
        "Could not prepare Chrome for Kaleido. "
        "Set BROWSER_PATH to your Chrome/Chromium binary, or run plotly_get_chrome."
    )


def install_chrome_cli():
    """
    Small entrypoint for a dedicated console script:
      poetry run telf-install-chrome
    """
    ensure_chrome_for_kaleido()


# ------------------------------
# Orchestration
# ------------------------------

def run_post_install_commands(
    gpu: bool = False,
    hpc: bool = False,
    hpc_conda: bool = False,
    gpu_toolkit: bool = False,
    skip_models: bool = False,
    kaleido_chrome: bool = False,
):
    """
    Execute post-install steps in a safe, idempotent way.
    """
    # 0) Kaleido/Chrome (optional but recommended if you need PNG/PDF export)
    if kaleido_chrome:
        ensure_chrome_for_kaleido()

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
        description="Post installation script for TELF (models/data, optional GPU/HPC extras, and Kaleido/Chrome setup)."
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
    p.add_argument(
        "--kaleido-chrome",
        action="store_true",
        help="Ensure Chrome is available for plotly+kaleido static image export.",
    )

    args = p.parse_args()
    try:
        run_post_install_commands(
            gpu=args.gpu,
            hpc=args.hpc,
            hpc_conda=args.hpc_conda,
            gpu_toolkit=args.gpu_toolkit,
            skip_models=args.skip_models,
            kaleido_chrome=args.kaleido_chrome,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}:\n  {' '.join(e.cmd)}")
        sys.exit(e.returncode)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
