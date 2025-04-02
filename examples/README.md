## Modules

### TELF.factorization

|         **Method**        |      **Dense**     |     **Sparse**     |       **GPU**      |       **CPU**      | **Multiprocessing** |       **HPC**      |                          **Description**                         | **Example** |
|:-------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:-------------------:|:------------------:|:----------------------------------------------------------------:|:-----------:|
|            NMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |              NMF with Automatic Model Determination                              |   [Link](examples/NMFk/NMFk.ipynb)  |
|        Custom NMFk        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                Use Custom NMF Functions with NMFk                                |   [Link](examples/NMFk/Custom_NMF_NMFk.ipynb)  |
|          TriNMFk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: |                    | NMF with Automatic Model Determination for Clusters and Patterns                 |   [Link](examples/TriNMFk/TriNMFk.ipynb)  |
|          RESCALk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |             RESCAL with Automatic Model Determination                            |   [Link](examples/RESCALk/RESCALk.ipynb)  |
|           RNMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         Recommender NMFk                                         |   [Link](examples/RNMFk/RNMFk.ipynb)  |
|           SymNMFk         | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         NMFk with Symmetric Clustering                           |   [Link](examples/SymNMFk/SymNMFk.ipynb)          |
|           WNMFk           | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         NMFk with weighting - used for recommendation system     |   [Link](examples/WNMFk/WNMFk.ipynb)          |
|           HNMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         Hierarchical NMFk                                        |   [Link](examples/HNMFk/HNMFk.ipynb)       |
|           BNMFk           | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                           Boolean NMFk                                           |   [Link](examples/BNMFk/BNMFk.ipynb) |
|           LMF             | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |                     |                    |                           Logistic Matrix Factorization                          |   [Link](examples/LMF/LMF.ipynb) |
|         SPLIT             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                     |        Joint NMFk factorization of multiple data via SPLIT                       | [Link](examples/SPLIT/00-SPLIT.ipynb) |
| SPLITTransfer | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  |                    |      Supervised transfer learning method via SPLIT and NMFk                      | [Link](examples/SPLITTransfer/00-SPLITTransfer.ipynb) |

### TELF.pre_processing

| **Method** | **Multiprocessing** |       **HPC**       |                           **Description**                          | **Example** |
|:----------:|:-------------------:|:-------------------:|:------------------------------------------------------------------:|:-----------:|
|   Vulture  | :heavy_check_mark:  | :heavy_check_mark:  |         Advanced text processing tool for cleaning and NLP         |  [Link](examples/Vulture)  |
|   Beaver   | :heavy_check_mark:  | :heavy_check_mark:  |        Fast matrix and tensor building tool for text mining        |  [Link](examples/Beaver)  |
|  iPenguin  | :heavy_check_mark:  |                     |         Online information retrieval tool for Scopus, SemanticScholar, and OSTI         | [Link](examples/iPenguin) |
|    Orca    | :heavy_check_mark:  |                     | Duplicate author detector for text mining and information retrieval |   [Link](examples/Orca)          |

### TELF.post_processing

| **Method** |                       **Description**                      | **Example** |
|:----------:|:----------------------------------------------------------:|:-----------:|
|    Wolf    |              Graph centrality and ranking tool             |      [Link](examples/Wolf)       |
|   Peacock  | Data visualization and generation of actionable statistics |  [Link](examples/Peacock) |
|    SeaLion    |              Generic report generation tool            | [Link](examples/SeaLion) |
|    Fox    |              Report generation tool for text data from NMFk using OpenAI            | [Link](examples/Fox)  |
|    ArcticFox    |        Report generation tool for text data from HNMFk using local LLMs            | [Link](examples/ArcticFox)  |

### TELF.applications

| **Method** |                            **Description**                           | **Example** |
|:----------:|:--------------------------------------------------------------------:|:-----------:|
|   Cheetah  |                        Fast search by keywords and phrases                       |    [Link](examples/Cheetah)         |
|    Bunny   | Dataset generation tool for documents and their citations/references |  [Link](examples/Bunny)  |
|  Penguin   |         Text storage tool                                    | [Link](examples/Penguin) |
|    Termite   | Knowladge graph building tool | :soon: |