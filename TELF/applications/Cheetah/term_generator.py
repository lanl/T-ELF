import os
import re
import numpy as np
from TELF.pre_processing import Beaver
from sklearn.feature_extraction.text import TfidfVectorizer

class SearchTermGenerator:
    def __init__(self, 
                 top_n=10, 
                 top_support=10,
                 top_word_as_support=False,
                ):
        self.top_n = top_n
        self.top_support = top_support
        self.top_word_as_support = top_word_as_support


    def process_dataframe(self, df, vocabulary, text_column='clean_title_abstract', reverse_subs=None, output_md_path=None):
        beaver = Beaver()
        co_matrix, _ = beaver.cooccurrence_matrix(
            dataset=df,
            target_column=text_column,
            cooccurrence_settings={"n_jobs": -1, "window_size": 100, "vocabulary": vocabulary},
            sppmi_settings={},
            save_path=None
        )
        dense_matrix = co_matrix.toarray() if hasattr(co_matrix, "toarray") else co_matrix

        vectorizer = TfidfVectorizer(token_pattern=r'\w+', vocabulary={w: i for i, w in enumerate(vocabulary)})
        tfidf_matrix = vectorizer.fit_transform(df[text_column].tolist())
        inv_vocab = {i: t for t, i in vectorizer.vocabulary_.items()}

        token_to_supports = {}
        aggregated_top_tokens = set()

        for i, row in df.iterrows():
            row_vec = tfidf_matrix[i].toarray()[0]
            sorted_idxs = np.argsort(row_vec)[::-1]

            row_tokens = []
            for j in sorted_idxs:
                if len(row_tokens) >= self.top_n or row_vec[j] <= 0:
                    break
                token = inv_vocab[j]
                row_tokens.append(token)

            row_text_set = set(re.findall(r'\w+', row[text_column].lower()))
            for token in row_tokens:
                aggregated_top_tokens.add(token)
                idx = vectorizer.vocabulary_.get(token)
                token_vec = dense_matrix[idx]
                candidates = []

                for j in np.argsort(token_vec)[::-1]:
                    if j == idx or token_vec[j] == 0:
                        continue
                    cand = vocabulary[j]
                    if cand in row_text_set:
                        candidates.append(cand)
                    if len(candidates) >= self.top_support:
                        break

                support_set = set(candidates)
                token_to_supports[token] = token_to_supports.get(token, support_set) & support_set

        if output_md_path:
            os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
            with open(output_md_path, 'w', encoding='utf-8') as f:
                for token in sorted(aggregated_top_tokens):
                    support_words = sorted(token_to_supports.get(token, []))
                    if reverse_subs:
                        token = reverse_subs.get(token, token)
                        support_words = [reverse_subs.get(w, w) for w in support_words]
                    f.write(f"## {token}\n")
                    f.write("positive: " + ", ".join(support_words) + "\n")
                    f.write("negative:\n")

        return sorted(aggregated_top_tokens), token_to_supports
    