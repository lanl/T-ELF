import argparse
import subprocess

def run_post_install_commands(gpu=False, hpc=False, hpc_conda=False, gpu_toolkit=False):

    if hpc and hpc_conda:
        print("Both HPC pip and HPC conda were True. Defaulting to hpc via pip.")
        hpc_conda = False
    
    print("Downloading SpaCy en_core_web_lg model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    print("Downloading SpaCy en_core_web_trf model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_trf"])
    print("Downloading NLTK wordnet and omw-1.4 data...")
    subprocess.run(["python", "-m", "nltk.downloader", "wordnet", "omw-1.4"])

    if gpu:
        print("Installing Cupy...")
        subprocess.run(["conda", "install", "-c", "conda-forge", "cupy", "numpy==2.0.0"])
    if hpc:
        print("Installing mpi4py via pip")
        subprocess.run(["pip", "install", "mpi4py"])
    if hpc_conda:
        print("Installing mpi4py via conda-forge")
        subprocess.run(["conda", "install", "-c", "conda-forge", "mpi4py"])
    if gpu_toolkit:
        print("Installing cudnn and cudatoolkit")
        subprocess.run(["conda", "install", "cudatoolkit"])
        subprocess.run(["conda", "install", "cudnn"])

    # correct the numpy version
    subprocess.run(["pip", "install", "numpy==2.0"])

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Post installation script for downloading models and data.")

    # Add arguments to control whether specific downloads happen
    parser.add_argument('--gpu', action='store_true', help="Install Cupy if using GPU")
    parser.add_argument('--hpc', action='store_true', help="Install mpi4py if using HPC using pip")
    parser.add_argument('--hpc-conda', action='store_true', help="Install mpi4py if using HPC using conda-forge")
    parser.add_argument('--gpu-toolkit', action='store_true', help="Install cudatoolkit and cudnn that may be needed in some systems.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    run_post_install_commands(
        gpu=args.gpu,
        hpc=args.hpc,
        hpc_conda=args.hpc_conda,
        gpu_toolkit=args.gpu_toolkit
    )