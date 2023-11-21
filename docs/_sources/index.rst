.. TELF documentation master file, created by
   sphinx-quickstart on Mon Oct  9 15:18:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tensor Extraction of Latent Features (T-ELF)
=================================================

.. image:: ../tensorsrnd.png
   :width: 648px
   :height: 400px
   :scale: 100 %
   :align: center


T-ELF presents an array of customizable software solutions crafted for analysis of datasets. Acting as a comprehensive toolbox, T-ELF specializes in data pre-processing, extraction of latent features, and structuring results to facilitate informed decision-making. Leveraging high-performance computing and cutting-edge GPU architectures, our toolbox is optimized for analyzing extensive datasets.

Central to T-ELF's core capabilities lie non-negative matrix and tensor factorization solutions for discovering multi-faceted hidden details in data, featuring automated model determination facilitating the estimation of latent factors or rank. This pivotal functionality ensures precise data modeling and the extraction of concealed patterns. Additionally, our software suite incorporates cutting-edge modules for both pre-processing and post-processing of data, tailored for diverse tasks including text mining, Natural Language Processing, and robust tools for matrix and tensor analysis and construction.

T-ELF's adaptability spans across a multitude of disciplines, positioning it as a robust AI and data analytics solution. Its proven efficacy extends across various fields such as Large-scale Text Mining, High Performance Computing, Computer Security, Applied Mathematics, Dynamic Networks and Ranking, Biology, Material Science, Medicine, Chemistry, Data Compression, Climate Studies, Relational Databases, Data Privacy, Economy, and Agriculture.


Resources
========================================
* `Examples <https://github.com/lanl/T-ELF/tree/main/examples>`_
* `Website <https://smart-tensors.lanl.gov>`_
* `Code <https://github.com/lanl/T-ELF>`_


Installation
========================================
**Option 1: Install via PIP and CONDA**

* **Step 1:** Install the library

.. code-block:: shell

   git clone https://github.com/lanl/T-ELF.git
   cd T-ELF
   conda create --name TELF python=3.11.5
   source activate TELF # or conda activate TELF
   python setup.py install

* **Step 2:** Install Spacy NLP model and NLTK Packages

.. code-block:: shell

   python -m spacy download en_core_web_lg
   python -m nltk.downloader wordnet omw-1.4

* **Step 3:** Install Cupy if using GPU (*Optional*)

.. code-block:: shell

   conda install -c conda-forge cupy

* **Step 4:** Install MPI if using HPC (*Optional*)

.. code-block:: shell

   module load <openmpi> # On a HPC Node
   pip install mpi4py


**Option 2: Install via CONDA Only**

* **Step 1:** Download the Library

.. code-block:: shell

   git clone https://github.com/lanl/T-ELF.git
   cd T-ELF

* **Step 2:** Install/Setup the Environment (CPU or GPU)
    - 2a: CPU

    .. code-block:: shell

      conda env create --file environment_cpu.yml

    - 2b: or GPU

    .. code-block:: shell

      conda env create --file environment_gpu.yml

* **Step 3:** Setup TELF

.. code-block:: shell

   conda activate TELF_conda
   conda develop .

- **Step 4:** Install Spacy NLP model and NLTK Packages

.. code-block:: shell

   python -m spacy download en_core_web_lg
   python -m nltk.downloader wordnet omw-1.4

* **Step 5:** Install MPI if using HPC (*Optional*)

.. code-block:: shell

   module load <openmpi>
   conda install -c conda-forge mpi4py


Capabilities
========================================

.. image:: ../Second_image_TensorNetworks.jpg
   :width: 450px
   :height: 400px
   :scale: 100 %
   :align: center

Please see our `publications <https://smart-tensors.lanl.gov/publications/>`_ for the capabilities.


How to Cite T-ELF?
========================================

**APA:**

.. code-block:: console

   Eren, M., Solovyev, N., Barron, R., Bhattarai, M., Boureima, I., Skau, E., Rasmussen, K., & Alexandrov, B. (2023). Tensor Extraction of Latent Features (T-ELF) (Version 0.0.2) [Computer software]. https://github.com/lanl/T-ELF

**BibTeX:**

.. code-block:: console

   @software{Tensor_Extraction_of_2023,
      author = {Eren, Maksim and Solovyev, Nick and Barron, Ryan and Bhattarai, Manish and Boureima, Ismael and Skau, Erik and Rasmussen, Kim and Alexandrov, Boian},
      month = oct,
      title = {{Tensor Extraction of Latent Features (T-ELF)}},
      url = {https://github.com/lanl/T-ELF},
      version = {0.0.2},
      year = {2023}
   }

Authors
========================================
- `Maksim E Eren <mailto:maksim@lanl.gov>`_: Advanced Research in Cyber Systems, Los Alamos National Laboratory
- `Nicholas Solovyev <mailto:nks@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Ryan Barron <mailto:barron@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Manish Bhattarai <mailto:ceodspspectrum@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Ismael Boureima <mailto:iboureima@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Erik Skau <mailto:ewskau@lanl.gov>`_: Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- `Kim Rasmussen <mailto:kor@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Boian Alexandrov <mailto:boian@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory

Copyright Notice
========================================
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL C Number: C22048**


License
========================================
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
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

Developer Test Suite
========================================
Developer test suites are located under ``tests/`` directory (located `here <https://github.com/lanl/T-ELF/tree/main/tests>`_).

Tests can be ran from this folder using ``python -m pytest *``.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   NMFk
   RESCALk
   TriNMFk
   Beaver
   Vulture
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
