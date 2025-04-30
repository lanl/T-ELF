# NM Law Data Pipeline

This example accompanies the research presented in our [paper](https://arxiv.org/abs/2502.20364) titled *"Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization"*.

**Abstract:** Agentic Generative AI, powered by Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), Knowledge Graphs (KGs), and Vector Stores (VSs), represents a transformative technology applicable to specialized domains such as legal systems, research, recommender systems, cybersecurity, and global security, including proliferation research. This technology excels at inferring relationships within vast unstructured or semi-structured datasets. The legal domain here comprises complex data characterized by extensive, interrelated, and semi-structured knowledge systems with complex relations. It comprises constitutions, statutes, regulations, and case law. Extracting insights and navigating the intricate networks of legal documents and their relations is crucial for effective legal research. Here, we introduce a generative AI system that integrates RAG, VS, and KG, constructed via Non-Negative Matrix Factorization (NMF), to enhance legal information retrieval and AI reasoning and minimize hallucinations. In the legal system, these technologies empower AI agents to identify and analyze complex connections among cases, statutes, and legal precedents, uncovering hidden relationships and predicting legal trends-challenging tasks that are essential for ensuring justice and improving operational efficiency. Our system employs web scraping techniques to systematically collect legal texts, such as statutes, constitutional provisions, and case law, from publicly accessible platforms like Justia. It bridges the gap between traditional keyword-based searches and contextual understanding by leveraging advanced semantic representations, hierarchical relationships, and latent topic discovery. This framework supports legal document clustering, summarization, and cross-referencing, for scalable, interpretable, and accurate retrieval for semi-structured data while advancing computational law and AI.

**The system:**
The goal is to minimize hallucination in LLMs, improve legal reasoning accuracy, and support scalable, domain-specific AI tools for computational law.
- Scrapes legal sources (statutes, constitutions, court opinions),
- Structures and cleans text data,
- Evaluates generative model outputs (e.g., GPT-4o, Gemini Pro, SMART-SLIC),
- Builds latent semantic topic models,
- And visualizes results across time, topic, and legal relevance.
  
## Citation
**APA:**
```latex
Barron, R., Eren, M.E., Serafimova, O.M., Matuszek, C., and Alexandrov, B.. Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization. In ICAIL ’25: 20th International Conference on Artificial Intelligence and Law, Jun. 16-20, 2025, Chicago, Illinois, USA. 10 pages.
```

**BibTeX:**
```latex
@article{Barron2025BridgingLK,
  title={Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization},
  author={Ryan Barron and Maksim Ekin Eren and Olga M. Serafimova and Cynthia Matuszek and Boian Alexandrov},
  journal={ArXiv},
  year={2025},
  volume={abs/2502.20364},
  url={https://api.semanticscholar.org/CorpusID:276647492}
}
```

---

##   00_data_collection

This directory is responsible for scraping and formatting legal data.

### Contents:

- **`00_dl_data.ipynb`** – Notebook for downloading or scraping data.
- **`format_csv.py`** – Script for formatting scraped data into structured CSVs.
- **`nm_scraper.py`** – Scraper module for collecting statutes, constitutional provisions, and court decisions from Justia.

---

##   02_benchmarking

This folder contains tools for evaluating LLM outputs against the legal dataset.

### Contents:

- **`llm_output_eval.py`** – Evaluation logic for single-model outputs.
- **`multi_model_qeval.py`** – Runs evaluation for multiple models side-by-side.
- **`qa_eval.py`** – Question-answer-based model benchmarking.

---

##   02_hnmfk_operation

This section contains code for topic modeling via Hierarchical Nonnegative Matrix Factorization (HNMFk).

### Contents:

- **`03-0_doc_word_matrix.ipynb`** – Generates document-word co-occurrence matrices.
- **`03-1_run_hnmfk.py`** – Executes the HNMFk pipeline and saves outputs.

---

##   03_visualizations

Visualization modules for model performance, topic breakdowns, and document-level insights.

### Contents:

- **`analysis_25_question.py`** – Visual analysis focused on 25 benchmark legal questions.
- **`analysis_58_question.py`** – Expanded benchmark across 58 legal questions.
- **`case_history.py`** – Visualizes outputs in a historical/legal context.
- **`estoppel_KG.py`** – Builds and visualizes a knowledge graph from estoppel-related outputs.
- **`Visualization_Examples.ipynb`** – Examples of various visualizations for model evaluation results.

---

# Evaluation Metadata

The legal question sets, created and reviewed by a lawyer subject matter expert, used for the paper can be found in the data directory ([here](../../../data/NM_law_questions_and_dates)). These datasets used to evaluate the TELF pipeline and the concepts introduced in the paper.

### [25_system_domain_questions.txt](../../../data/NM_law_questions_and_dates/25_system_domain_questions.txt)
- A plain text file listing 25 system-generated legal questions.
- These are used to evaluate how well models generalize across legal domains.

### [58_sme.txt](../../../data/NM_law_questions_and_dates/58_sme.txt)
- A manually curated set of 58 legal benchmark questions.
- Crafted by subject-matter experts (SMEs) to cover a range of doctrines, statutes, and case law.
- Useful for high-precision model evaluation in legal QA tasks.

### [court_data.json](../../../data/NM_law_questions_and_dates/court_data.json)
- Structured metadata mapping NM Supreme Court and Court of Appeals decision counts per year as well as expansion events.


###   Example Use Cases

- Use `58_sme.txt` to prompt LLMs and measure factually supported responses.
- Align model predictions with `court_data.json` to validate whether answers respect procedural/legal timelines.
- Fine-tune models on `25_system_domain_questions.txt` for zero-shot or few-shot evaluations.