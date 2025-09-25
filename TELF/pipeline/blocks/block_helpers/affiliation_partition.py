import ast
import json
from pathlib import Path
import pandas as pd

UNKNOWN_COUNTRY = "unknown"

def _parse_affiliations_with_country(raw) -> list[tuple[str, str]]:
    """
    Parse a JSON/Python-literal dict into [(name, country), …].
    Always returns a country (missing ⇒ 'unknown'). Returns [] if unparseable.
    Expected shape (examples):
      '{"0": {"name": "MIT", "country": "United States"}}'
      '{0: {"name": "LANL"}}'
    """
    if raw is None:
        return []
    try:
        if pd.isna(raw):  # safe even if raw isn't a pandas scalar
            return []
    except Exception:
        pass

    if isinstance(raw, dict):
        parsed = raw
    else:
        s = str(raw).strip()
        if s in ("", "{}", "[]"):
            return []
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return []

    if not isinstance(parsed, dict):
        return []

    out: list[tuple[str, str]] = []
    for info in parsed.values():
        if not isinstance(info, dict):
            continue
        name = info.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        country = info.get("country")
        if isinstance(country, str):
            country = country.strip() or UNKNOWN_COUNTRY
        elif country is None:
            country = UNKNOWN_COUNTRY
        else:
            country = str(country).strip() or UNKNOWN_COUNTRY
        out.append((name.strip(), country))
    return out

def generate_top_affiliations_with_country(
    df_path: str | Path,
    affils_output_path: str | Path,
    min_total_papers: int = 20,
    country_filter: str | None = None,           # ← filter to exactly this country (or 'unknown')
    partition_by_year: bool = False,             # ← also emit one CSV per year
    per_year_output_dir: str | Path | None = None,
):
    """
    Reads df_path (must have 'year' and 'affiliations'), computes per-(affiliation, country, year)
    paper counts for affiliations whose total_papers (within the current filter) ≥ min_total_papers.
    Always includes a 'country' value; if missing in the source, uses 'unknown'.

    If country_filter is provided, restricts to that single country (exact string match, including 'unknown').
    """
    df = pd.read_csv(df_path)

    if 'year' not in df.columns:
        raise KeyError("Expected a 'year' column.")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year'].notna()].copy()
    df['year'] = df['year'].astype(int)

    if 'affiliations' not in df.columns:
        raise KeyError("Expected an 'affiliations' column.")

    # 1) Parse affiliations → list of (name, country or 'unknown')
    df_aff = df.copy()
    df_aff['affil_tuples'] = df_aff['affiliations'].apply(_parse_affiliations_with_country)

    # 2) Explode into one row per (paper, affiliation_name, country)
    exploded_aff = df_aff.explode('affil_tuples')
    exploded_aff = exploded_aff[exploded_aff['affil_tuples'].notna()].copy()
    exploded_aff[['affiliation_name', 'country']] = pd.DataFrame(
        exploded_aff['affil_tuples'].tolist(), index=exploded_aff.index
    )
    # Defensive fill (should already be set by parser)
    exploded_aff['country'] = exploded_aff['country'].fillna(UNKNOWN_COUNTRY)

    # 3) Optional: restrict to one specific country
    if country_filter is not None:
        exploded_aff = exploded_aff[exploded_aff['country'] == country_filter].copy()

    # 4) Totals per affiliation (within current filter scope)
    total_per_aff = (
        exploded_aff
        .groupby(['affiliation_name', 'country'])
        .size()
        .reset_index(name='total_papers')
        .sort_values('total_papers', ascending=False)
    )

    print("=== Affiliations",
          f"in [{country_filter}]" if country_filter is not None else "(all countries)",
          "with their total paper counts ===")
    print(total_per_aff.head(20).to_string(index=False))
    print("───────────────────────────────────────────────────────────────────────────\n")

    # 5) Keep affiliations with ≥ min_total_papers
    top_affils = total_per_aff[total_per_aff['total_papers'] >= min_total_papers][
        ['affiliation_name', 'country']
    ]

    if top_affils.empty:
        print(f"No affiliation{' in ' + country_filter if country_filter else ''} "
              f"meets ≥ {min_total_papers} total papers.")
        aff_year_counts = pd.DataFrame(columns=['affiliation_name','country','year','paper_count'])
    else:
        # 6) Per-year counts for top affiliations
        exploded_aff_top = exploded_aff.merge(top_affils, on=['affiliation_name','country'], how='inner')
        aff_year_counts = (
            exploded_aff_top
            .groupby(['affiliation_name','country','year'])
            .size()
            .reset_index(name='paper_count')
            .sort_values(['affiliation_name','country','year'])
            .reset_index(drop=True)
        )

    # 7) Write consolidated CSV
    affils_output_path = Path(affils_output_path)
    affils_output_path.parent.mkdir(parents=True, exist_ok=True)
    if affils_output_path.suffix == "":
        affils_output_path = affils_output_path.with_suffix(".csv")
    aff_year_counts.to_csv(affils_output_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(aff_year_counts)} rows to {affils_output_path} "
          f"(≥ {min_total_papers} papers"
          f"{', country=' + country_filter if country_filter is not None else ', all countries'})")

    # 8) Optional: one file per year (same columns)
    if partition_by_year and not aff_year_counts.empty:
        out_dir = Path(per_year_output_dir) if per_year_output_dir else affils_output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = affils_output_path.stem
        suffix = affils_output_path.suffix or ".csv"
        for yr in sorted(aff_year_counts['year'].unique()):
            yr_df = aff_year_counts[aff_year_counts['year'] == yr]
            yr_path = out_dir / f"{stem}.year={yr}{suffix}"
            yr_df.to_csv(yr_path, index=False, encoding="utf-8-sig")
            print(f"→ Wrote {len(yr_df)} rows for year {yr} to {yr_path}")

# # All countries in output (missing → 'unknown'), consolidated CSV only
# generate_top_affiliations_with_country(
#     "papers.csv", "out/affiliations_top.csv", min_total_papers=20
# )

# # Only the United States (others excluded), plus per-year files
# generate_top_affiliations_with_country(
#     "papers.csv", "out/affiliations_top.csv",
#     min_total_papers=10,
#     country_filter="United States",
#     partition_by_year=True,
#     per_year_output_dir="out/by_year"
# )

# # Only entries whose country was missing in the source (now labeled 'unknown')
# generate_top_affiliations_with_country(
#     "papers.csv", "out/affiliations_unknown.csv",
#     min_total_papers=5,
#     country_filter="unknown"
# )
