import os
import warnings
import pandas as pd
from .cheetah import Cheetah

class CheetahTermFormatter:
    """
    Loads search terms from a Markdown file and returns them as
    plain strings or dict blocks, with optional category filtering.
    Can also generate a substitutions lookup dict mapping phrases
    to underscored forms and back, if substitutions=True.
    
    New parameters:
      all_categories (bool): if True, ignore `category` and
        `include_general` and include every section.
    """
    def __init__(self, markdown_file, lower=False, category=None,
                 include_general=True, substitutions=False, all_categories=False):
        self.markdown_file    = markdown_file
        self.lower            = lower
        self.category         = category
        self.include_general  = include_general
        self.substitutions    = substitutions
        self.all_categories   = all_categories

        self.substitution_forward = {}
        self.substitution_reverse = {}

        # parse the markdown into self.terms
        self.terms = self._parse_markdown()

        # optionally build lookup table
        if self.substitutions:
            self._build_substitutions_lookup()


    def _parse_markdown(self):
        terms = []
        current_term = None
        positives = []
        negatives = []
        active_block = False
        current_section = None

        try:
            with open(self.markdown_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            warnings.warn(f"File '{self.markdown_file}' not found. Returning empty list.")
            return []

        for raw in lines:
            line = raw.strip()

            # Section header
            if line.startswith("# Category:"):
                current_section = line.split(":", 1)[1].strip()
                continue

            # Decide whether to include this section
            if self.all_categories:
                include_section = True
            elif self.category is None:
                # no filtering → include everything
                include_section = True
            else:
                if current_section is None and self.include_general:
                    include_section = True
                else:
                    include_section = (current_section == self.category)

            # Term header
            if line.startswith("##"):
                # finish previous block
                if current_term is not None and active_block:
                    if positives or negatives:
                        terms.append({
                            current_term: {
                                "positives": positives,
                                "negatives": negatives
                            }
                        })
                    else:
                        terms.append(current_term)

                # reset for new block
                positives = []
                negatives = []
                header = line.lstrip("#").strip()
                if self.lower:
                    header = header.lower()
                current_term = header
                active_block = include_section

            # collect positives / negatives
            elif active_block and line.lower().startswith("must have:"):
                items = [i.strip() for i in line.split(":", 1)[1].split(",") if i.strip()]
                positives.extend(items)
            elif active_block and line.lower().startswith("exclude with:"):
                items = [i.strip() for i in line.split(":", 1)[1].split(",") if i.strip()]
                negatives.extend(items)

        # final block
        if current_term is not None and active_block:
            if positives or negatives:
                terms.append({
                    current_term: {
                        "positives": positives,
                        "negatives": negatives
                    }
                })
            else:
                terms.append(current_term)

        return terms

    def _build_substitutions_lookup(self):
        """
        Build a dict mapping each term to its underscored form and vice versa.
        """
        for entry in self.terms:
            if isinstance(entry, str):
                term = entry
                underscored = term.replace(" ", "_")
                self.substitution_forward[term] = underscored
                self.substitution_reverse[underscored] = term
            elif isinstance(entry, dict):
                for term in entry.keys():
                    underscored = term.replace(" ", "_")
                    self.substitution_forward[term] = underscored
                    self.substitution_reverse[underscored] = term

    def get_terms(self):
        return self.terms

    def get_substitution_maps(self):
        """
        Return the substitutions lookup dict (empty if substitutions=False).
        """
        return self.substitution_forward, self.substitution_reverse


def convert_txt_to_cheetah_markdown(txt_path, markdown_path):
    import ast

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    markdown_lines = []

    for line in lines:
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = ast.literal_eval(line)
                for key, value in parsed.items():
                    positives = [v.lstrip('+') for v in value if v.startswith('+')]
                    negatives = [v for v in value if not v.startswith('+')]
                    markdown_lines.append(f"## {key}")
                    if positives:
                        markdown_lines.append(f"positives: {', '.join(positives)}")
                    if negatives:
                        markdown_lines.append(f"negatives: {', '.join(negatives)}")
            except Exception as e:
                print(f"Skipping line due to parse error: {line}\nError: {e}")
        else:
            markdown_lines.append(f"## {line.strip()}")

    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))

    print(f"Converted markdown saved to: {markdown_path}")
