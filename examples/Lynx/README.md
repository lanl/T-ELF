## Usage:
1. ```cd``` into the root directory of this repository

    ```cd TELF```
2. Create a ```projects``` directory.

    ```mkdir projects```
3. Put your post-processing folder for each project under this directory.

    ```cp -r /path/to/project1 projects```
4. On terminal start the server

    ```streamlit run TELF/applications/Lynx/frontend/main.py```
5. **Optional** if running on a remote server, forward the ports by running the following on the local terminal 

    ```ssh USER@HOST -L 8501:localhost:8501```
6. Have fun!



## OPTIONAL STARTUP

zsh ./start_lynx.sh  -p  ../Full\ TELF\ Pipeline/single_block_examples/example_results/semantic_HNMFk_collection_slurm_option/07_SemanticHNMFk


## ViralTensors Example

Rapid discovery of efficient medical countermeasures (MCM) to emerging pathogens is key to successful management of disease outbreaks and pandemic prevention. We perform comprehensive analysis of global systems data representing host-virus molecular interaction networks to discover therapeutics acting broadly as antiviral MCM. We apply a new Artificial Intelligence (AI) platform, ViralTensors, based on algorithms utilizing nonnegative matrix and tensor factorization (SmartTensors). ViralTensors extract from global systems biological data explainable latent (not directly observable) features in the host cells that are essential for virus replication. Based on these features, our tool integrates molecular interaction networks from transcriptome and virus-host interaction proteome data to identify cellular wiring patterns caused by virus replication and to predict drugs that target pathways specific to virus infection mechanisms. With this framework, we identify cellular processes, including endosome trafficking, gap junction signaling, and cholesterol synthesis, as host pathways universally required for RNA and DNA virus replication. Inhibition of these pathways with FDA approved drugs suppressed the replication efficiency of broad-spectrum virus species, thus validating the AI performance in the identification of essential and explainable patters from multi-omics datasets for development of virus agnostic therapeutics. (LA-UR-24-24060)

We have provided example link prediction of PPI data with ViralTensors. Follow the below steps after installing T-ELF to use Lynx for exploring this data:
1. ```cd``` into the root directory of this repository

    ```cd TELF```

2. Create a ```projects``` directory.

    ```mkdir projects```

3. Put your post-processing folder for each project under this directory.

    ```cp -r data/lynx/ViralTensors_v1 projects/.```

4. On terminal start the server

    ```streamlit run TELF/applications/Lynx/frontend/main.py```