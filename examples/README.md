## Modules

### TELF.factorization

|         **Method**        |      **Dense**     |     **Sparse**     |       **GPU**      |       **CPU**      | **Multiprocessing** |       **HPC**      |                          **Description**                         | **Example** |
|:-------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:-------------------:|:------------------:|:----------------------------------------------------------------:|:-----------:|
|            NMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |              NMF with Automatic Model Determination                              |   [Link](NMFk/NMFk.ipynb)  |
|        Custom NMFk        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                Use Custom NMF Functions with NMFk                                |   [Link](NMFk/Custom_NMF_NMFk.ipynb)  |
|          TriNMFk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: |                    | NMF with Automatic Model Determination for Clusters and Patterns                 |   [Link](TriNMFk/TriNMFk.ipynb)  |
|          RESCALk          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |             RESCAL with Automatic Model Determination                            |   [Link](RESCALk/RESCALk.ipynb)  |
|           RNMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         Recommender NMFk                                         |   [Link](RNMFk/RNMFk.ipynb)  |
|           SymNMFk         | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         NMFk with Symmetric Clustering                           |   [Link](SymNMFk/SymNMFk.ipynb)          |
|           WNMFk           | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         NMFk with weighting - used for recommendation system     |   [Link](WNMFk/WNMFk.ipynb)          |
|           HNMFk           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                         Hierarchical NMFk                                        |   [Link](HNMFk/HNMFk.ipynb)       |
|           BNMFk           | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: |                           Boolean NMFk                                           |   [Link](BNMFk/BNMFk.ipynb) |
|           LMF             | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |                     |                    |                           Logistic Matrix Factorization                          |   [Link](LMF/LMF.ipynb) |
|         SPLIT             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                     |        Joint NMFk factorization of multiple data via SPLIT                       | [Link](SPLIT/00-SPLIT.ipynb) |
| SPLITTransfer | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  |                    |      Supervised transfer learning method via SPLIT and NMFk                      | [Link](SPLITTransfer/00-SPLITTransfer.ipynb) |

### TELF.pre_processing

| **Method** | **Multiprocessing** |       **HPC**       |                           **Description**                          | **Example** |
|:----------:|:-------------------:|:-------------------:|:------------------------------------------------------------------:|:-----------:|
|   Vulture  | :heavy_check_mark:  | :heavy_check_mark:  |         Advanced text processing tool for cleaning and NLP         |  [Link](Vulture)  |
|   Beaver   | :heavy_check_mark:  | :heavy_check_mark:  |        Fast matrix and tensor building tool for text mining        |  [Link](Beaver)  |
|  iPenguin  | :heavy_check_mark:  |                     |         Online information retrieval tool for Scopus, SemanticScholar, and OSTI         | [Link](iPenguin) |
|    Orca    | :heavy_check_mark:  |                     | Duplicate author detector for text mining and information retrieval |   [Link](Orca)          |

### TELF.post_processing

| **Method** |                       **Description**                      | **Example** |
|:----------:|:----------------------------------------------------------:|:-----------:|
|    Wolf    |              Graph centrality and ranking tool             |      [Link](Wolf)       |
|   Peacock  | Data visualization and generation of actionable statistics |  [Link](Peacock) |
|    SeaLion    |              Generic report generation tool            | [Link](SeaLion) |
|    Fox    |              Report generation tool for text data from NMFk using OpenAI            | [Link](Fox)  |
|    ArcticFox    |        Report generation tool for text data from HNMFk using local LLMs            | [Link](ArcticFox)  |

### TELF.applications

| **Method** |                            **Description**                           | **Example** |
|:----------:|:--------------------------------------------------------------------:|:-----------:|
|   Cheetah  |                        Fast search by keywords and phrases                       |    [Link](Cheetah)         |
|    Bunny   | Dataset generation tool for documents and their citations/references |  [Link](Bunny)  |
|  Penguin   |         Text storage tool                                    | [Link](Penguin) |
|    Termite   | Knowladge graph building tool | :soon: |


## Use Cases

| **Example** |                            **Description**                           | **Link** |
|:----------:|:--------------------------------------------------------------------:|:-----------:|
|   NM Law Data           |                        Domain specific data for AI and RAG system written in our  [paper](https://arxiv.org/abs/2502.20364) about New Mexico Law that uses the TELF pipeline       |  [Link](NM%20Law%20Data)|
|    Full TELF Pipeline   | An end-to-end pipeline demonstration, from collection to analysis |  :soon:  |