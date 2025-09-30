from __future__ import annotations
# TELF/pre_processing/Orca/orca.py
import os
import ast
import copy
import pickle
import warnings
import pandas as pd
import networkx as nx
from tqdm import tqdm
# Minimal no-op AuthorMatcher fallback
# Path 1 (temporary to match your current import resolution):
#   TELF/pipeline/blocks/AuthorMatcher.py
# Path 2 (correct long-term location):
#   TELF/pre_processing/Orca/AuthorMatcher.py

import pandas as pd
from typing import Dict, Iterable, List, Tuple, Union

class AuthorMatcher:
    """
    Fallback stub used when the real AuthorMatcher isn't available.
    Produces an empty matches DataFrame (or builds rows from known_matches if you provide them).
    This is enough for Orca to proceed via the Scopus-only/S2-only residual logic.
    """

    def __init__(self, df: pd.DataFrame, n_jobs: int = -1, verbose: bool = False):
        self.df = df
        self.n_jobs = n_jobs
        self.verbose = verbose

    def match(self, known_matches: Dict[str, Union[str, Iterable[str]]] | None = None) -> pd.DataFrame:
        cols = ["S2_Author_ID", "S2_Author_Name", "SCOPUS_Author_ID"]
        if not known_matches:
            # No matches known → return empty, Orca will handle residual mapping.
            return pd.DataFrame(columns=cols)

        # Build a quick s2_id -> name map if available
        s2_name_map: Dict[str, str] = {}
        try:
            if {"s2_author_ids", "s2_authors"}.issubset(self.df.columns):
                tmp = self.df[["s2_author_ids", "s2_authors"]].dropna(how="any")
                for ids, names in zip(tmp["s2_author_ids"], tmp["s2_authors"]):
                    ids = str(ids).split(";")
                    names = str(names).split(";")
                    for i, n in zip(ids, names):
                        s2_name_map.setdefault(i, n)
        except Exception:
            pass

        rows: List[Dict[str, str]] = []
        for k, vs in known_matches.items():
            if not isinstance(vs, (list, tuple, set)):
                vs = [vs]
            for v in vs:
                k_str, v_str = str(k), str(v)

                # Heuristics to assign which side is S2 vs Scopus (best effort)
                k_in_s2 = k_str in s2_name_map
                v_in_s2 = v_str in s2_name_map
                if k_in_s2 and not v_in_s2:
                    s2_id, scopus_id = k_str, v_str
                elif v_in_s2 and not k_in_s2:
                    s2_id, scopus_id = v_str, k_str
                else:
                    # Ambiguous → skip quietly
                    continue

                rows.append(
                    {
                        "S2_Author_ID": s2_id,
                        "S2_Author_Name": s2_name_map.get(s2_id, "Unknown"),
                        "SCOPUS_Author_ID": scopus_id,
                    }
                )

        return pd.DataFrame(rows, columns=cols)


class Orca:
    """
    Construct SLIC author ids + apply them to a SLIC-style paper dataframe.
    """

    def __init__(self, duplicates=None, s2_duplicates=None, verbose=False):
        self.slic_df = None
        self.duplicates = duplicates
        self.s2_duplicates = s2_duplicates
        self.verbose = verbose

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, df, scopus_duplicates=None, s2_duplicates=None, known_matches=None, n_jobs=-1):
        """
        Form the SLIC map from Scopus-only, S2-only, or hybrid dataframes.
        """
        if scopus_duplicates is not None:
            self.duplicates = scopus_duplicates
        if s2_duplicates is not None:
            self.s2_duplicates = s2_duplicates

        has_scopus = {"author_ids", "authors"}.issubset(df.columns)
        has_s2 = {"s2_author_ids", "s2_authors"}.issubset(df.columns)

        if has_scopus and not has_s2:
            self.slic_df = self._run_scopus(df)
            return self.slic_df
        if has_s2 and not has_scopus:
            self.slic_df = self._run_s2_only(df)
            return self.slic_df
        if not (has_scopus or has_s2):
            raise ValueError(
                "Orca.run(): DataFrame must contain Scopus ('author_ids','authors') "
                "or S2 ('s2_author_ids','s2_authors') columns."
            )

        # Hybrid path
        affiliations_map = self.__generate_affiliations_map(df)
        s2_author_map = self.__generate_author_map(df, "s2_author_ids", "s2_authors")
        scopus_author_map = self.__generate_author_map(df, "author_ids", "authors")

        known_matches = {} if not known_matches else known_matches
        am = AuthorMatcher(df, n_jobs=n_jobs, verbose=self.verbose)
        am_df = am.match(known_matches=known_matches)

        # Enrich with known Scopus duplicates
        am_enriched = self.__add_scopus_duplicates(am_df, self.duplicates)
        am_df = pd.concat([am_df, am_enriched], axis=0, ignore_index=True)

        # Build components across S2/Scopus ids
        slic_count = 0
        seen_s2, seen_scopus = set(), set()
        matches = self.__uncouple_author_matches(am_df, self.s2_duplicates)

        slic_df = {
            "slic_id": [],
            "slic_name": [],
            "scopus_ids": [],
            "scopus_names": [],
            "scopus_affiliations": [],
            "s2_ids": [],
            "s2_names": [],
        }

        # 1) Matched S2<->Scopus groups
        for entry in matches:
            s2_id_set = entry["s2"]
            s2_name_set = {s2_author_map.get(x, "Unknown") for x in s2_id_set if x in s2_author_map}
            scopus_id_set = entry["scopus"]
            scopus_name_set = {scopus_author_map[x] for x in scopus_id_set if x in scopus_author_map}
            scopus_affiliations = self.__merge_scopus_affiliations(scopus_id_set, affiliations_map)
            slic_name = entry["name"]

            seen_s2 |= s2_id_set
            seen_scopus |= scopus_id_set

            slic_df["slic_id"].append(f"S{slic_count}")
            slic_df["slic_name"].append(slic_name)
            slic_df["scopus_ids"].append(";".join(sorted(scopus_id_set)))
            slic_df["scopus_names"].append(";".join(sorted(scopus_name_set)))
            slic_df["scopus_affiliations"].append(scopus_affiliations)
            slic_df["s2_ids"].append(";".join(sorted(s2_id_set)))
            slic_df["s2_names"].append(";".join(sorted(s2_name_set)))
            slic_count += 1

        # 2) Scopus-only residuals
        if "author_ids" in df.columns:
            df_scopus_authors = {x for y in df.author_ids.to_list() if not pd.isna(y) for x in y.split(";")}
            df_scopus_authors -= seen_scopus
            for scopus_id in sorted(df_scopus_authors):
                scopus_name = scopus_author_map.get(scopus_id, None)
                slic_df["slic_id"].append(f"S{slic_count}")
                slic_df["slic_name"].append(scopus_name)
                slic_df["scopus_ids"].append(scopus_id)
                slic_df["scopus_names"].append(scopus_name)
                slic_df["scopus_affiliations"].append(affiliations_map.get(scopus_id, None))
                slic_df["s2_ids"].append(None)
                slic_df["s2_names"].append(None)
                slic_count += 1

        # 3) S2-only residuals
        if "s2_author_ids" in df.columns:
            df_s2_authors = {x for y in df.s2_author_ids.to_list() if not pd.isna(y) for x in y.split(";")}
            df_s2_authors -= seen_s2
            for s2_id in sorted(df_s2_authors):
                slic_df["slic_id"].append(f"S{slic_count}")
                slic_df["scopus_ids"].append(None)
                slic_df["scopus_names"].append(None)
                slic_df["scopus_affiliations"].append(None)

                s2_dup_ids = self.s2_duplicates.get(s2_id)
                if s2_dup_ids is not None:
                    s2_ids_all = {s2_id} | set(s2_dup_ids)
                    s2_author_map_local = {x: s2_author_map.get(x, "Unknown") for x in s2_ids_all}
                    s2_name_set = {v for v in s2_author_map_local.values() if v != "Unknown"}
                    slic_name = max(s2_name_set, key=len) if s2_name_set else None

                    slic_df["slic_name"].append(slic_name)
                    slic_df["s2_ids"].append(";".join(sorted(s2_ids_all)) if s2_ids_all else None)
                    slic_df["s2_names"].append(";".join(sorted(s2_name_set)) if s2_name_set else None)
                    seen_s2 |= s2_ids_all
                else:
                    s2_name = s2_author_map.get(s2_id, None)
                    slic_df["slic_name"].append(s2_name)
                    slic_df["s2_ids"].append(s2_id)
                    slic_df["s2_names"].append(s2_name)
                    seen_s2.add(s2_id)

                slic_count += 1

        slic_df = pd.DataFrame.from_dict(slic_df)
        slic_df = slic_df.loc[slic_df.slic_name != "Unknown"].copy().reset_index(drop=True)
        self.slic_df = slic_df
        return slic_df

    def apply(self, df, slic_df=None):
        """
        Apply the SLIC id mapping to a SLIC papers dataframe.
        Keeps papers even when SLIC author ids are missing (warns only).
        """
        if slic_df is None and self.slic_df is None:
            return ValueError("No SLIC ID map found. First, compute the map with Orca.run()")
        if slic_df is not None and self.slic_df is not None:
            warnings.warn(
                "[Orca]: slic_df was passed as an argument however this Orca object already has a "
                "stored slic_df object.\n\t\tOverwriting stored slic_df with given argument. If this "
                "message is unexpected, use Orca.apply() without specifying `slic_df`",
                RuntimeWarning,
            )

        if slic_df is not None:
            self.slic_df = slic_df

        # Uniqueness guards
        if "eid" in df.columns and df.eid.nunique() != len(df.loc[~df.eid.isnull()]):
            df = df[~df["eid"].duplicated(keep="first") | df["eid"].isna()].copy()
            warnings.warn("[Orca]: Encountered duplicate Scopus IDs (`eid`) in df. Dropping duplicate papers.")
        if "s2id" in df.columns and df.s2id.nunique() != len(df.loc[~df.s2id.isnull()]):
            df = df[~df["s2id"].duplicated(keep="first") | df["s2id"].isna()].copy()
            warnings.warn("[Orca]: Encountered duplicate S2 IDs (`s2id`) in df. Dropping duplicate papers.")

        # Compute per-source SLIC ids
        if "s2id" in df.columns and "eid" in df.columns:
            scopus_df = self.__compute_slic_scopus(df)
            s2_df = self.__compute_slic_s2(df)
            df2 = pd.merge(df, scopus_df, on="eid", how="outer")
            df3 = pd.merge(df2, s2_df, on="s2id", how="outer")
            orca_df = df3.copy()

            # unify slic_author_ids
            orca_df["slic_author_ids"] = orca_df["slic_author_ids_x"].combine_first(orca_df["slic_author_ids_y"])
            orca_df = orca_df.drop(columns=["slic_author_ids_x", "slic_author_ids_y"])

            # unify slic_affiliations if present as suffixes
            if "slic_affiliations_x" in orca_df.columns or "slic_affiliations_y" in orca_df.columns:
                left = orca_df.get("slic_affiliations_x")
                right = orca_df.get("slic_affiliations_y")
                if left is not None and right is not None:
                    orca_df["slic_affiliations"] = left.combine_first(right)
                    orca_df.drop(
                        columns=[c for c in ["slic_affiliations_x", "slic_affiliations_y"] if c in orca_df.columns],
                        inplace=True,
                    )
                elif left is not None:
                    orca_df.rename(columns={"slic_affiliations_x": "slic_affiliations"}, inplace=True)
                else:
                    orca_df.rename(columns={"slic_affiliations_y": "slic_affiliations"}, inplace=True)

        elif "s2id" in df.columns:
            s2_df = self.__compute_slic_s2(df)
            orca_df = pd.merge(df, s2_df, on="s2id", how="outer")
            if "slic_affiliations" not in orca_df.columns:
                orca_df["slic_affiliations"] = None

        else:
            scopus_df = self.__compute_slic_scopus(df)
            orca_df = pd.merge(df, scopus_df, on="eid", how="outer")
            if "slic_affiliations_x" in orca_df.columns:
                orca_df.rename(columns={"slic_affiliations_x": "slic_affiliations"}, inplace=True)
                if "slic_affiliations_y" in orca_df.columns:
                    orca_df.drop(columns=["slic_affiliations_y"], inplace=True)

        # Keep papers with missing SLIC ids (warn only)
        if "slic_author_ids" in orca_df.columns:
            missing = int(orca_df["slic_author_ids"].isna().sum())
            if missing:
                warnings.warn(f"[Orca]: {missing} papers have no SLIC author IDs (S2-only or unmatched). Keeping them.")

        if "slic_affiliations" not in orca_df.columns:
            orca_df["slic_affiliations"] = None

        # Add SLIC author names
        slic_authors = {k: v for k, v in zip(self.slic_df.slic_id.to_list(), self.slic_df.slic_name.to_list())}

        def map_ids_to_names(ids):
            if pd.isna(ids):
                return None
            names = [slic_authors.get(str(i), "") for i in ids.split(";")]
            names = [name for name in names if name]
            return ";".join(names)

        orca_df["slic_authors"] = orca_df["slic_author_ids"].apply(map_ids_to_names)
        return orca_df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _run_scopus(self, df):
        df = df.copy()
        for col in ("author_ids", "authors"):
            if col not in df.columns:
                df[col] = pd.NA

        affiliations_map = self.__generate_affiliations_map(df)
        scopus_author_map = self.__generate_author_map(df, "author_ids", "authors")

        duplicates = {}
        for entry in self.duplicates:
            if not (set(scopus_author_map.keys()) & set(entry)):
                continue

            entry = entry.copy()
            best_id = sorted(entry, key=lambda x: len(scopus_author_map.get(x, "")), reverse=True)[0]
            merged_affiliations = self.__merge_scopus_affiliations(entry, affiliations_map)
            affiliations_map[best_id] = merged_affiliations

            entry.remove(best_id)
            for x in entry:
                if x in scopus_author_map:
                    del scopus_author_map[x]
                if x in affiliations_map:
                    del affiliations_map[x]

            duplicates[best_id] = list(entry)

        slic_df = {
            "slic_id": [],
            "slic_name": [],
            "scopus_ids": [],
            "scopus_names": [],
            "scopus_affiliations": [],
            "s2_ids": [],
            "s2_names": [],
        }

        for i, scopus_id in enumerate(scopus_author_map):
            scopus_name = scopus_author_map.get(scopus_id)
            scopus_affiliations = affiliations_map.get(scopus_id)
            if scopus_id in duplicates:
                scopus_id = ";".join([scopus_id] + duplicates[scopus_id])

            slic_df["slic_id"].append(f"S{i}")
            slic_df["slic_name"].append(scopus_name)
            slic_df["scopus_ids"].append(scopus_id)
            slic_df["scopus_names"].append(scopus_name)
            slic_df["scopus_affiliations"].append(scopus_affiliations)
            slic_df["s2_ids"].append(None)
            slic_df["s2_names"].append(None)

        return pd.DataFrame.from_dict(slic_df)

    def _run_s2_only(self, df):
        df = df.copy()
        for col in ("s2_author_ids", "s2_authors"):
            if col not in df.columns:
                df[col] = pd.NA

        s2_author_map = self.__generate_author_map(df, "s2_author_ids", "s2_authors")

        visited = set()
        groups = []

        for s2_id in s2_author_map.keys():
            if s2_id in visited:
                continue
            group = {s2_id}
            if s2_id in self.s2_duplicates:
                group |= set(self.s2_duplicates[s2_id])
            visited |= group
            groups.append(group)

        for root, dups in self.s2_duplicates.items():
            if root not in visited:
                group = {root} | set(dups)
                visited |= group
                groups.append(group)

        rows = {
            "slic_id": [],
            "slic_name": [],
            "scopus_ids": [],
            "scopus_names": [],
            "scopus_affiliations": [],
            "s2_ids": [],
            "s2_names": [],
        }

        idx = 0
        seen = set()
        for g in groups:
            name_set = {s2_author_map.get(x, "Unknown") for x in g if x in s2_author_map}
            if name_set == {"Unknown"}:
                name_set = set()
            slic_name = max(name_set, key=len) if name_set else None

            rows["slic_id"].append(f"S{idx}")
            rows["slic_name"].append(slic_name)
            rows["scopus_ids"].append(None)
            rows["scopus_names"].append(None)
            rows["scopus_affiliations"].append(None)
            rows["s2_ids"].append(";".join(sorted(g)))
            rows["s2_names"].append(";".join(sorted(name_set)) if name_set else None)

            seen |= g
            idx += 1

        for s2_id, s2_name in s2_author_map.items():
            if s2_id in seen:
                continue
            rows["slic_id"].append(f"S{idx}")
            rows["slic_name"].append(s2_name if s2_name != "Unknown" else None)
            rows["scopus_ids"].append(None)
            rows["scopus_names"].append(None)
            rows["scopus_affiliations"].append(None)
            rows["s2_ids"].append(s2_id)
            rows["s2_names"].append(s2_name)
            idx += 1

        slic_df = pd.DataFrame.from_dict(rows)
        slic_df = slic_df.loc[slic_df.slic_name.notna()].reset_index(drop=True)
        return slic_df

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def __add_scopus_duplicates(self, auth_df, duplicates):
        out_df = pd.DataFrame(columns=auth_df.columns)
        all_df_ids = set(auth_df.SCOPUS_Author_ID.to_list())
        if self.verbose:
            print("[Orca]: Scanning for Scopus duplicates in dataset. . .")
        for scopus_ids in tqdm(duplicates, total=len(duplicates), disable=not self.verbose):
            if not scopus_ids & all_df_ids:
                continue
            tmp_df = auth_df.loc[auth_df.SCOPUS_Author_ID.isin(scopus_ids)]
            if len(tmp_df.SCOPUS_Author_ID.unique()) > 1:
                row = tmp_df.iloc[0].copy()
                for scopus_id in tmp_df.SCOPUS_Author_ID.unique():
                    if row.SCOPUS_Author_ID == scopus_id:
                        continue
                    new_row = row.copy()
                    new_row.SCOPUS_Author_ID = scopus_id
                    out_df = pd.concat([out_df, new_row.to_frame().T], ignore_index=True)
        return out_df

    def __add_s2_duplicates(self, duplicates):
        out = {}
        seen = set()
        for s in duplicates:
            if any(elem in seen for elem in s):
                raise ValueError(
                    "Detected multiple entries for s2 duplicates across sets. "
                    "Make sure that all known duplicates are constrained to a single set"
                )
            seen.update(s)
            if len(s) == 1:
                continue
            for element in s:
                out[element] = [x for x in s if x != element]
        return out

    def __propagate_duplicates(self, a_map, a_duplicates):
        a_map_update = {}
        for a_id, b_ids in a_map.items():
            if a_id in a_duplicates:
                for dup_a_id in a_duplicates[a_id]:
                    a_map_update.setdefault(dup_a_id, set()).update(set(b_ids))
        for dup_a_id in a_map_update:
            a_map_update[dup_a_id] = list(a_map_update[dup_a_id])
        a_map.update(a_map_update)

    def __uncouple_author_matches(self, auth_df, s2_duplicates):
        s2_map = auth_df.groupby("S2_Author_ID")["SCOPUS_Author_ID"].agg(set).to_dict()
        scopus_map = auth_df.groupby("SCOPUS_Author_ID")["S2_Author_ID"].agg(set).to_dict()
        self.__propagate_duplicates(s2_map, s2_duplicates)

        s2_map = {f"B_{k}": {f"A_{x}"} if isinstance(v, str) else {f"A_{x}" for x in v} for k, v in s2_map.items()}
        scopus_map = {f"A_{k}": {f"B_{x}"} if isinstance(v, str) else {f"B_{x}" for x in v} for k, v in scopus_map.items()}

        s2_name_map = auth_df.groupby("S2_Author_ID")["S2_Author_Name"].agg(lambda x: max(x, key=len)).to_dict()

        G = nx.DiGraph()
        for k, v_set in scopus_map.items():
            for v in v_set:
                G.add_edge(k, v)
        for k, v_set in s2_map.items():
            for v in v_set:
                G.add_edge(k, v)

        components = list(nx.weakly_connected_components(G))
        matches = []
        for component_set in components:
            mdict = {"scopus": set(), "s2": set()}
            for c in component_set:
                (mdict["scopus"] if c.startswith("A_") else mdict["s2"]).add(c[2:])
            str_gen = ((pid, s2_name_map[pid]) for pid in mdict["s2"] if pid in s2_name_map)
            _, name = max(str_gen, key=lambda x: len(x[1]), default=(None, "Unknown"))
            mdict["name"] = name
            matches.append(mdict)
        return matches

    def __generate_author_map(self, df, id_col, name_col):
        if self.verbose:
            print(f"[Orca]: Generating {id_col}-{name_col} map. . .")
        if id_col not in df.columns or name_col not in df.columns:
            return {}
        auth_map = {}
        tmp = df[[id_col, name_col]].dropna(how="any")
        if tmp.empty:
            return {}
        for id_list, auth_list in tqdm(
            zip(tmp[id_col].to_list(), tmp[name_col].to_list()), total=len(tmp), disable=not self.verbose
        ):
            if not isinstance(id_list, str) or not isinstance(auth_list, str):
                continue
            for auth_id, name in zip(id_list.split(";"), auth_list.split(";")):
                if auth_id and auth_id not in auth_map:
                    auth_map[auth_id] = name
        return auth_map

    def __compute_slic_scopus(self, df):
        tmp_df = df.loc[~df["eid"].isnull()]
        scopus_authors, scopus_affiliations = {}, {}

        for eid, author_ids, affiliations in zip(
            tmp_df["eid"].to_list(), tmp_df["author_ids"].to_list(), tmp_df["affiliations"].to_list()
        ):
            if not pd.isna(author_ids):
                scopus_authors[eid] = author_ids
            if not pd.isna(affiliations):
                if isinstance(affiliations, str):
                    affiliations = ast.literal_eval(affiliations)
                scopus_affiliations[eid] = affiliations

        scopus_df = {"eid": [], "slic_author_ids": [], "slic_affiliations": []}
        scopus_to_slic = {
            x: k
            for k, v in zip(self.slic_df.slic_id.to_list(), self.slic_df.scopus_ids.to_list())
            if not pd.isna(v)
            for x in v.split(";")
        }

        missing_authors = set()
        for eid in tmp_df.eid.to_list():
            slic_author_ids = []
            author_ids = scopus_authors.get(eid)
            if author_ids is not None:
                for scopus_id in author_ids.split(";"):
                    scopus_id = str(scopus_id)
                    slic_id = scopus_to_slic.get(scopus_id)
                    if slic_id is None:
                        missing_authors.add(scopus_id)
                    else:
                        slic_author_ids.append(str(slic_id))

            aff_dict, del_dict = {}, []
            affiliations = scopus_affiliations.get(eid)
            if affiliations is not None:
                for aff_id, aff_info_shallow in affiliations.items():
                    if isinstance(aff_info_shallow, list):
                        continue
                    del_list = []
                    aff_info = copy.deepcopy(aff_info_shallow)

                    for i in range(len(aff_info["authors"])):
                        scopus_id = str(aff_info["authors"][i])
                        if scopus_id not in scopus_to_slic:
                            del_list.append(scopus_id)
                            missing_authors.add(scopus_id)
                        else:
                            aff_info["authors"][i] = scopus_to_slic[scopus_id]

                    for d in del_list:
                        for cand in (d, str(d)):
                            if cand in aff_info["authors"]:
                                aff_info["authors"].remove(cand)
                                break

                    if not aff_info["authors"]:
                        del_dict.append(aff_id)
                    aff_dict[aff_id] = aff_info

                for d in del_dict:
                    del aff_dict[d]

            scopus_df["eid"].append(eid)
            scopus_df["slic_author_ids"].append(";".join(slic_author_ids) if slic_author_ids else None)
            scopus_df["slic_affiliations"].append(aff_dict if aff_dict else None)

        if len(missing_authors) > 0:
            warnings.warn(
                f"[Orca]: {len(missing_authors)} Scopus IDs did not have corresponding SLIC ID and were removed"
            )

        return pd.DataFrame.from_dict(scopus_df)

    def __compute_slic_s2(self, df):
        tmp_df = df.loc[~df["s2id"].isnull()]
        s2_to_slic = {
            x: k
            for k, v in zip(self.slic_df.slic_id.to_list(), self.slic_df.s2_ids.to_list())
            if not pd.isna(v)
            for x in v.split(";")
        }

        s2_df = {"s2id": [], "slic_author_ids": []}
        missing_authors = set()
        for s2id, s2_author_ids in zip(tmp_df["s2id"].to_list(), tmp_df["s2_author_ids"].to_list()):
            slic_author_ids = []
            if not pd.isna(s2_author_ids):
                for s2_auth_id in s2_author_ids.split(";"):
                    if s2_auth_id not in s2_to_slic:
                        missing_authors.add(s2_auth_id)
                    else:
                        slic_author_ids.append(s2_to_slic[s2_auth_id])

            s2_df["s2id"].append(s2id)
            s2_df["slic_author_ids"].append(";".join(slic_author_ids) if slic_author_ids else None)

        if len(missing_authors) > 0:
            warnings.warn(
                f"[Orca]: {len(missing_authors)} S2 IDs did not have corresponding SLIC ID and were removed"
            )

        return pd.DataFrame.from_dict(s2_df)

    def __load_pickle(self, fn):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return pickle.load(open((os.path.join(current_dir, "data", fn)), "rb"))

    def __generate_affiliations_map(self, df):
        """
        Build a map SCOPUS_AUTHOR_ID -> affiliation info with paper provenance.
        Paper id priority: eid > s2id > doi > synthetic row id.
        """
        import pandas as pd, ast

        if "affiliations" not in df.columns or "year" not in df.columns:
            return {}

        id_priority = ("eid", "s2id", "doi")
        pid_col = next((c for c in id_priority if c in df.columns), None)
        if pid_col is None:
            df = df.copy()
            df["_orca_row_id"] = [f"row_{i}" for i in range(len(df))]
            pid_col = "_orca_row_id"

        affiliations_map = {}
        pid_series = df[pid_col].astype(object).where(pd.notna(df[pid_col]), None)
        year_series = df["year"]
        aff_series = df["affiliations"]

        for pid, year, affiliations in zip(pid_series.tolist(), year_series.tolist(), aff_series.tolist()):
            if pd.isna(affiliations):
                continue
            if isinstance(affiliations, str):
                affiliations = ast.literal_eval(affiliations)

            year = 0 if pd.isna(year) else int(year)
            paper_id = None if pid is None else str(pid)

            for aff_id, info in affiliations.items():
                if isinstance(info, list):
                    continue
                aff_name = info.get("name", "Unknown")
                aff_country = info.get("country", "Unknown")

                for auth_id in info.get("authors", []):
                    if auth_id not in affiliations_map:
                        affiliations_map[auth_id] = {}
                    if aff_id not in affiliations_map[auth_id]:
                        affiliations_map[auth_id][aff_id] = {
                            "name": aff_name,
                            "country": aff_country,
                            "first_seen": year,
                            "last_seen": year,
                            "papers": set(),
                        }

                    entry = affiliations_map[auth_id][aff_id]
                    if paper_id is not None:
                        entry["papers"].add(paper_id)
                    if entry["first_seen"] in (0, None) or (year and year < entry["first_seen"]):
                        entry["first_seen"] = year
                    if entry["last_seen"] in (0, None) or (year and year > entry["last_seen"]):
                        entry["last_seen"] = year

        for auth_id in affiliations_map:
            for aff_id, aff_info in affiliations_map[auth_id].items():
                aff_info["papers"] = list(aff_info["papers"])
                if not aff_info["first_seen"]:
                    aff_info["first_seen"] = None
                if not aff_info["last_seen"]:
                    aff_info["last_seen"] = None

        return affiliations_map

    def __merge_scopus_affiliations(self, scopus_ids, data):
        merged = {}
        if not data or not scopus_ids:
            return None

        all_keys = set()
        for sid in scopus_ids:
            if sid in data:
                all_keys.update(data[sid].keys())

        for key in all_keys:
            items = [data[sid][key] for sid in scopus_ids if sid in data and key in data[sid]]
            if not items:
                continue

            names = [it.get("name", "Unknown") for it in items if it.get("name") not in (None, "Unknown")]
            countries = [it.get("country", "Unknown") for it in items if it.get("country") not in (None, "Unknown")]

            paper_sets = []
            for it in items:
                p = it.get("papers", [])
                if isinstance(p, set):
                    paper_sets.append(p)
                elif p is None:
                    continue
                else:
                    paper_sets.append(set(p))
            merged_papers = sorted(set().union(*paper_sets)) if paper_sets else []

            first_vals = [it.get("first_seen") for it in items if isinstance(it.get("first_seen"), int)]
            last_vals = [it.get("last_seen") for it in items if isinstance(it.get("last_seen"), int)]

            merged[key] = {
                "name": names[0] if names else "Unknown",
                "country": countries[0] if countries else "Unknown",
                "first_seen": min(first_vals) if first_vals else "Unknown",
                "last_seen": max(last_vals) if last_vals else "Unknown",
                "papers": merged_papers,
            }

        return merged or None

    # ─────────────────────────────────────────────────────────────────────────
    # Getters / Setters
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def duplicates(self):
        return self._duplicates

    @duplicates.setter
    def duplicates(self, duplicates):
        if duplicates is None:
            self._duplicates = []
        elif isinstance(duplicates, list):
            self._duplicates = duplicates
        else:
            raise TypeError(f"{type(duplicates)} is an invalid type for `duplicates`")

    @property
    def s2_duplicates(self):
        return self._s2_duplicates

    @s2_duplicates.setter
    def s2_duplicates(self, s2_duplicates):
        if s2_duplicates is None:
            self._s2_duplicates = {}
        elif isinstance(s2_duplicates, list):
            self._s2_duplicates = self.__add_s2_duplicates(s2_duplicates)
        else:
            raise TypeError(f"{type(s2_duplicates)} is an invalid type for `s2_duplicates`")
