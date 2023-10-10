# Tensor Extraction of Latent Features (T-ELF) <img align="left" width="50" height="50" src="docs/cube.jpg">

<div align="center", style="font-size: 50px">

[![Build Status](https://github.com/lanl/T-ELF/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/lanl/T-ELF/actions/workflows/ci_tests.yml/badge.svg?branch=main) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg) [![Python Version](https://img.shields.io/badge/python-v3.8.5-blue)](https://img.shields.io/badge/python-v3.8.5-blue)

</div>

<p align="center">
  <img src="docs/tensorsrnd.png">
</p>

**T-ELF** delivers a comprehensive suite of software solutions tailored for in-depth analysis of vast datasets, accurate and precise extraction of hidden patterns, harnessing the power of high-performance computing and cutting-edge GPU architectures. Our approach is underpinned by scalable and highly efficient algorithms.

<div align="center", style="font-size: 50px">

### [:information_source: Documentation](https://lanl.github.io/telf/) &emsp; [:orange_book: Examples](examples/) &emsp; [:page_with_curl: Publications](Publications.md) &emsp; [:link: Website](https://smart-tensors.LANL.gov)

</div>


## Installation

#### Option 1: Install via PIP and CONDA
- **Step 1:** Install the library
```shell
git clone https://github.com/lanl/T-ELF.git
cd T-ELF
conda create --name TELF python=3.8.5
source activate TELF # or conda activate TELF
python setup.py install
```
- **Step 2:** Install Spacy NLP model and NLTK Packages
```shell
python -m spacy download en_core_web_lg
python -m nltk.downloader wordnet omw-1.4
```
- **Step 3:** Install Cupy if using GPU (*Optional*)
```shell
conda install -c conda-forge cupy
```
- **Step 4:** Install MPI if using HPC (*Optional*)
```shell
module load <openmpi> # On a HPC Node
pip install mpi4py
```

#### Option 2: Install via CONDA Only
- **Step 1:** Download the Library
```shell
git clone https://github.com/lanl/T-ELF.git
cd T-ELF
```
- **Step 2:** Install/Setup the Environment (CPU or GPU)
    - 2a: CPU
    ```shell
    conda env create --file environment_cpu.yml
    ```
    - 2b: or GPU
    ```shell
    conda env create --file environment_gpu.yml
    ```
- **Step 3:** Setup TELF
```shell
conda activate TELF_conda
conda develop .
```
- **Step 4:** Install Spacy NLP model and NLTK Packages
```shell
python -m spacy download en_core_web_lg
python -m nltk.downloader wordnet omw-1.4
```
- **Step 5:** Install MPI if using HPC (*Optional*)
```shell
module load <openmpi>
conda install -c conda-forge mpi4py
```

## Capabilities

<div align="center", style="font-size: 50px">

<p align="center">
  <img src="docs/Second_image_TensorNetworks.jpg">
</p>

### [:page_with_curl: Publications](Publications.md)

</div>


## Modules

### TELF.factorization

|         **Method**        |      **Dense**     |     **Sparse**     |       **GPU**      |       **CPU**      | **Multiprocessing** |       **HPC**      |                          **Description**                         | **Example** | **Release Status** |
|:-------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:-------------------:|:------------------:|:----------------------------------------------------------------:|:-----------:|:------------------:|
|            NMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |              NMF with Automatic Model Determination                              |   [Link](examples/NMFk/NMFk.ipynb)  | :white_check_mark: |
|        Custom NMFk        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                Use Custom NMF Functions with NMFk                                |   [Link](examples/NMFk/Custom_NMF_NMFk.ipynb)  | :white_check_mark: |
|          TriNMFk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: |                    | NMF with Automatic Model Determination for Clusters and Patterns                 |   [Link](examples/TriNMFk/TriNMFk.ipynb)  | :white_check_mark: |
|          RESCALk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: |                    |             RESCAL with Automatic Model Determination                            |   [Link](examples/RESCALk/RESCALk.ipynb)  | :white_check_mark: |
|           BNMFk           |                    |                    |                    |                    |                     |                    |                           Boolean NMFk                                           |             |       :soon:       |
|           HNMFk           |                    |                    |                    |                    |                     |                    |                         Hierarchical NMFk                                        |             |       :soon:       |
|           RNMFk           |                    |                    |                    |                    |                     |                    |                         Recommender NMFk                                         |             |       :soon:       |
|         SPLIT NMFk        |                    |                    |                    |                    |                     |                    |        Joint NMFk factorization of multiple data via SPLIT                       |             |       :soon:       |
| SPLIT Transfer Classifier |                    |                    |                    |                    |                     |                    |      Supervised transfer learning method via SPLIT and NMFk                      |             |       :soon:       |
|           CP-ALS          |                    |                    |                    |                    |                     |                    |    Alternating least squares algorithm for canonical polyadic decomposition      |             |       :soon:       |
|           CP-APR          |                    |                    |                    |                    |                     |                    |    Alternating Poisson regression algorithm for canonical polyadic decomposition |             |       :soon:       |
|           NTDS_FAPG       |                    |                    |                    |                    |                     |                    |                      Non-negative Tucker Tensor Decomposition                    |             |       :soon:       |
 
### TELF.pre_processing

| **Method** | **Multiprocessing** |       **HPC**       |                           **Description**                          | **Example** | **Release Status** |
|:----------:|:-------------------:|:-------------------:|:------------------------------------------------------------------:|:-----------:|:------------------:|
|   Vulture  | :heavy_check_mark: | :heavy_check_mark: |         Advanced text processing tool for cleaning and NLP         |  [Link](examples/Vulture)  | :white_check_mark: |
|   Beaver   | :heavy_check_mark: | :heavy_check_mark: |        Fast matrix and tensor building tool for text mining        |  [Link](examples/Beaver)  | :white_check_mark: |
|  iPenguin  |                     |                     |         Online Semantic Scholar information retrieval tool         |             |       :soon:       |
|    Orca    |                     |                     | Duplicate author detector for text mining and information retrival |             |       :soon:       |
|            |                     |                     |                                                                    |             |                    |

### TELF.post_processing

| **Method** |                       **Description**                      | **Example** | **Release Status** |
|:----------:|:----------------------------------------------------------:|:-----------:|:------------------:|
|   Peacock  | Data visualization and generation of actionable statistics |             |       :soon:       |
|    Wolf    |              Graph centrality and ranking tool             |             |       :soon:       |

### TELF.applications

| **Method** |                            **Description**                           | **Example** | **Release Status** |
|:----------:|:--------------------------------------------------------------------:|:-----------:|:------------------:|
|   Cheetah  |                        Fast search by keywords                       |             |       :soon:       |
|    Bunny   | Dataset generation tool for documents and their citations/references |             |       :soon:       |


## How to Cite T-ELF?
If you use T-ELF please cite.

**APA:**
>Eren, M., Solovyev, N., Barron, R., Bhattarai, M., Boureima, I., Skau, E., Rasmussen, K., & Alexandrov, B. (2023). Tensor Extraction of Latent Features (T-ELF) (Version 0.0.1) [Computer software]. https://github.com/lanl/T-ELF

**BibTeX:**
```latex
@software{Tensor_Extraction_of_2023,
author = {Eren, Maksim and Solovyev, Nick and Barron, Ryan and Bhattarai, Manish and Boureima, Ismael and Skau, Erik and Rasmussen, Kim and Alexandrov, Boian},
month = oct,
title = {{Tensor Extraction of Latent Features (T-ELF)}},
url = {https://github.com/lanl/T-ELF},
version = {0.0.1},
year = {2023}
}
```

## Authors
- [Maksim Ekin Eren](mailto:maksim@lanl.gov): Advanced Research in Cyber Systems, Los Alamos National Laboratory
- [Nicholas Solovyev](mailto:nks@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- [Ryan Barron](mailto:barron@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- [Manish Bhattarai](mailto:ceodspspectrum@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- [Ismael Boureima](mailto:iboureima@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- [Erik Skau](mailto:ewskau@lanl.gov): Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- [Kim Rasmussen](mailto:kor@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- [Boian S. Alexandrov](mailto:boian@lanl.gov): Theoretical Division, Los Alamos National Laboratory

## Patents
>Boian ALEXANDROV, o. S. F., New Mexico, Maksim Ekin EREN, of Sante Fe, New Mexico, Manish BHATTARAI, of Albuquerque, New Mexico, Kim Orskov RASMUSSEN of Sante Fe, New Mexico, and Charles K. NICHOLAS, of Columbia, Maryland, (“Assignor”) DATA IDENTIFICATION AND CLASSIFICATION METHOD, APPARATUS, AND SYSTEM. No. 63/472,188. Triad National Security, LLC. (June 9, 2023).

>BS. Alexandrov, LB. Alexandrov, and VG. Stanev et al. 2020. Source identification by non-negative matrix factorization combined with semi-supervised clustering. US Patent S10,776,718 (2020).

## Copyright Notice
>© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL C Number: C22048**

## License
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
 
2.Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
 
3.Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Developer Test Suite
Developer test suites are located under [```tests/```](tests/) directory. Tests can be ran from this folder using ```python -m pytest *```.