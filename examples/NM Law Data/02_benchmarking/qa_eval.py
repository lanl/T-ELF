import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import warnings
import torch
import csv
import re
import spacy

from rouge_score import rouge_scorer
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from summac.model_summac import SummaCConv


class LegalQuestionAIProcessor:
    def __init__(self, cleaned_output_csv, ai_output_dir, evaluated_output_csv):
        self.cleaned_output_csv = cleaned_output_csv
        self.ai_output_dir = ai_output_dir
        self.evaluated_output_csv = evaluated_output_csv
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_responses(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*overflowing tokens.*")

        df_q = pd.read_csv(self.cleaned_output_csv)
        df_r = pd.read_csv(os.path.join(self.ai_output_dir, "all_responses.csv"))
        df_r["question"] = df_r["question"].astype(str).str.strip().str.strip('"')
        df_q = df_q[df_q["question"] != "**"]
        df_r = df_r[df_r["question"] != "**"]
        df_merged = df_r.merge(df_q, on="question", how="inner", suffixes=("_resp", "_ques"))
        print(f"Ready to evaluate {len(df_merged)} rows...")

        print("Loading models...")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bleu_metric = evaluate.load("bleu")
        self.bertscore_metric = evaluate.load("bertscore")

        self.nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(self.device)

        self.factcc_tokenizer = BertTokenizer.from_pretrained("manueldeprada/FactCC")
        self.factcc_model = BertForSequenceClassification.from_pretrained("manueldeprada/FactCC").to(self.device)

        self.summaC = SummaCConv(
            models=["vitc"],
            bins="percentile",
            granularity="sentence",
            nli_labels="e",
            device=str(self.device),
            start_file="default",
            agg="mean",
        )

        self.spacy_nlp = spacy.load("en_core_web_sm")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(self._evaluate_single_row, row.question, row.model, row.answer, row.source)
                for row in df_merged.itertuples(index=False)
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[Error] Skipping row: {e}")

        with open(self.evaluated_output_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "question", "model", "answer", "source",
                "rouge1", "rouge2", "rougeL",
                "bleu", "bertscore_f1",
                "exact_match", "substring_match",
                "nli_entailment", "summaC_score", "factcc_score",
                "NE_prec", "Num_prec",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Evaluation complete. Results saved to {self.evaluated_output_csv}")

    def _evaluate_single_row(self, question, model, answer, source):
        rouge = self.rouge_scorer.score(answer, source)
        bleu = self.bleu_metric.compute(predictions=[answer], references=[[source]])["bleu"]
        bert_f1 = self.bertscore_metric.compute(predictions=[answer], references=[source], lang="en")["f1"][0]
        exact = int(answer.strip() == source.strip())
        substr = int(answer.strip() in source)

        nli_inputs = self.nli_tokenizer.encode_plus(source, answer, return_tensors="pt", truncation=True)
        nli_inputs = {k: v.to(self.device) for k, v in nli_inputs.items()}
        with torch.no_grad():
            logits = self.nli_model(**nli_inputs).logits
            entailment = torch.softmax(logits, dim=-1)[0][2].item()

        summaC_score = self.summaC.score([source], [answer], granularity="sentence")["scores"][0]
        factcc_score = self._compute_factcc_score(source, answer)
        ne_prec, num_prec = self._compute_entity_precision(source, answer)

        return {
            "question": question,
            "model": model,
            "answer": answer,
            "source": source,
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
            "bleu": bleu,
            "bertscore_f1": bert_f1,
            "exact_match": exact,
            "substring_match": substr,
            "nli_entailment": entailment,
            "summaC_score": summaC_score,
            "factcc_score": factcc_score,
            "NE_prec": ne_prec,
            "Num_prec": num_prec,
        }

    def _compute_factcc_score(self, source, answer):
        inputs = self.factcc_tokenizer(
            text=source,
            text_pair=answer,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.factcc_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs[0][1].item()

    def _compute_entity_precision(self, source, answer):
        source_ents = set(ent.text for ent in self.spacy_nlp(source).ents)
        answer_ents = set(ent.text for ent in self.spacy_nlp(answer).ents)
        ne_prec = (len(answer_ents & source_ents) / len(answer_ents)) if answer_ents else 1.0

        def extract_nums(text):
            return set(re.findall(r'\b\d+(?:,\d+)?(?:\.\d+)?\b', text))

        source_nums = extract_nums(source)
        answer_nums = extract_nums(answer)
        num_prec = (len(answer_nums & source_nums) / len(answer_nums)) if answer_nums else 1.0
        return ne_prec, num_prec

