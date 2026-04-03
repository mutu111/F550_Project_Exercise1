"""
Build sector-level sentiment indicators from transcript CSV files.

Inputs:
    outputs/tech_20_companies_transcripts_2024.csv
    outputs/industrials_20_companies_transcripts_2024.csv

Outputs:
    outputs/outlook_sentences.csv
    outputs/firm_quarter_sentiment.csv
    outputs/sector_sentiment.csv
    outputs/sentiment_results.csv
    outputs/transcript_sentiment_details.csv
    outputs/sector_sentiment_lexicon.png
    outputs/sector_sentiment_finbert.png
    outputs/sector_sentiment_combined.png
    outputs/firm_sentiment_boxplot.png
    outputs/avg_sector_sentiment_bar.png
    outputs/tech_company_quarter_heatmap_finbert.png
    outputs/industrials_company_quarter_heatmap_finbert.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

try:
    import torch
    from transformers import pipeline
except ModuleNotFoundError as exc:
    missing_pkg = exc.name or "transformers"
    raise SystemExit(
        "Missing dependency: "
        f"`{missing_pkg}`.\n\n"
        "Install required packages with:\n"
        "  pip install -r requirements.txt"
    ) from exc


TECH_PATH = Path("outputs/tech_20_companies_transcripts_2024.csv")
INDUSTRIALS_PATH = Path("outputs/industrials_20_companies_transcripts_2024.csv")
OUTPUT_DIR = Path("outputs")
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32

OUTLOOK_KEYWORDS_STRICT = [
    "guidance",
    "outlook",
    "expect",
    "expects",
    "expected",
    "forecast",
    "forecasting",
    "next quarter",
    "next year",
    "looking ahead",
    "we believe",
    "we anticipate",
]

LM_DICTIONARY_PATH = Path("data/LoughranMcDonald_MasterDictionary_2024.csv")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lm_dictionary(path: Path) -> tuple[set[str], set[str]]:
    """Load Loughran-McDonald positive and negative word lists."""
    if not path.exists():
        raise SystemExit(
            "Missing dictionary file: "
            f"`{path}`.\n\n"
            "Place `LoughranMcDonald_MasterDictionary_2024.csv` in the `data/` folder and rerun."
        )

    df = pd.read_csv(path)
    df["word"] = df["Word"].astype(str).str.lower()

    positive = set(df[df["Positive"] > 0]["word"])
    negative = set(df[df["Negative"] > 0]["word"])

    return positive, negative


LM_POSITIVE, LM_NEGATIVE = load_lm_dictionary(LM_DICTIONARY_PATH)


def load_and_combine_data() -> pd.DataFrame:
    """Load sector CSVs, align labels, and normalize transcript text."""
    tech_df = pd.read_csv(TECH_PATH)
    industrials_df = pd.read_csv(INDUSTRIALS_PATH)

    tech_df["sector"] = "Tech"
    industrials_df["sector"] = "Industrials"

    df = pd.concat([tech_df, industrials_df], ignore_index=True)
    df = df.dropna(subset=["transcript"]).copy()
    df["transcript"] = df["transcript"].astype(str)
    df["transcript_normalized"] = df["transcript"].str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quarter"] = df["quarter"].astype("Int64")
    df["year"] = df["year"].astype("Int64")
    return df


def split_into_sentences(text: str) -> List[str]:
    """Split transcript text into rough sentence units."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]


def build_sentence_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Explode transcript records to sentence-level rows."""
    rows = []
    columns = ["company", "symbol", "sector", "year", "quarter", "date", "sentence", "sentence_lower"]

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Splitting transcripts"):
        for sentence in split_into_sentences(row.transcript):
            rows.append(
                {
                    "company": row.company,
                    "symbol": row.symbol,
                    "sector": row.sector,
                    "year": row.year,
                    "quarter": row.quarter,
                    "date": row.date,
                    "sentence": sentence,
                    "sentence_lower": sentence.lower(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def is_outlook_sentence(sentence_lower: str) -> bool:
    """Check whether a sentence looks forward-looking."""
    return any(keyword in sentence_lower for keyword in OUTLOOK_KEYWORDS_STRICT)


def filter_outlook_sentences(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sentences that look forward-looking."""
    work = sentence_df.copy()
    work["is_outlook"] = work["sentence_lower"].apply(is_outlook_sentence)
    outlook_df = work[work["is_outlook"]].copy().reset_index(drop=True)
    return outlook_df


def tokenize_words(text: str) -> List[str]:
    """Simple tokenizer for lexicon sentiment scoring."""
    return re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())


def lexicon_score(sentence: str) -> float:
    """Compute a sentence-level lexicon sentiment score."""
    tokens = tokenize_words(sentence)
    if len(tokens) == 0:
        return np.nan

    pos = sum(token in LM_POSITIVE for token in tokens)
    neg = sum(token in LM_NEGATIVE for token in tokens)
    return (pos - neg) / len(tokens)


def build_finbert_pipeline():
    """Create the FinBERT text-classification pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=FINBERT_MODEL_NAME,
        tokenizer=FINBERT_MODEL_NAME,
        top_k=None,
        truncation=True,
        padding=True,
        device=device,
    )


def score_finbert_in_batches(sentences: Sequence[str], sentiment_pipe) -> List[float]:
    """Score outlook sentences with FinBERT in batches."""
    if not sentences:
        return []

    scores: List[float] = []
    total_batches = (len(sentences) + FINBERT_BATCH_SIZE - 1) // FINBERT_BATCH_SIZE

    for start in tqdm(range(0, len(sentences), FINBERT_BATCH_SIZE), total=total_batches, desc="Running FinBERT"):
        batch = [sentence.strip() for sentence in sentences[start:start + FINBERT_BATCH_SIZE]]
        outputs = sentiment_pipe(batch)

        for output in outputs:
            if isinstance(output, dict):
                output = [output]
            label_scores = {item["label"].lower(): item["score"] for item in output}
            scores.append(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0))

    return scores


def score_outlook_sentences(outlook_df: pd.DataFrame) -> pd.DataFrame:
    """Attach lexicon and FinBERT scores to the outlook sentence dataframe."""
    work = outlook_df.copy()
    work["sentiment_lexicon"] = work["sentence"].apply(lexicon_score)

    cleaned_sentences = work["sentence"].fillna("").astype(str).tolist()
    if cleaned_sentences:
        sentiment_pipe = build_finbert_pipeline()
        work["sentiment_finbert"] = score_finbert_in_batches(cleaned_sentences, sentiment_pipe)
    else:
        work["sentiment_finbert"] = np.nan

    return work


def aggregate_sentiment(outlook_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate sentiment across three levels:
    1. sentence
    2. firm-quarter
    3. sector-quarter
    """
    transcript_df = (
        outlook_df.groupby(["sector", "company", "symbol", "year", "quarter", "date"], as_index=False)
        .agg(
            outlook_sentence_count=("sentence", "size"),
            sentiment_lexicon=("sentiment_lexicon", "mean"),
            sentiment_finbert=("sentiment_finbert", "mean"),
            outlook_text=("sentence", lambda s: " ".join(s.astype(str))),
        )
        .sort_values(["sector", "symbol", "year", "quarter", "date"])
        .reset_index(drop=True)
    )

    firm_quarter_df = (
        transcript_df.groupby(["company", "symbol", "sector", "year", "quarter"], as_index=False)
        .agg(
            transcript_count=("outlook_text", "size"),
            outlook_sentence_count=("outlook_sentence_count", "sum"),
            sentiment_lexicon=("sentiment_lexicon", "mean"),
            sentiment_finbert=("sentiment_finbert", "mean"),
        )
        .sort_values(["sector", "symbol", "year", "quarter"])
        .reset_index(drop=True)
    )

    sector_df = (
        firm_quarter_df.groupby(["sector", "year", "quarter"], as_index=False)
        .agg(
            firm_count=("symbol", "nunique"),
            outlook_sentence_count=("outlook_sentence_count", "mean"),
            sentiment_lexicon=("sentiment_lexicon", "mean"),
            sentiment_finbert=("sentiment_finbert", "mean"),
        )
        .sort_values(["year", "quarter", "sector"])
        .reset_index(drop=True)
    )

    return transcript_df, firm_quarter_df, sector_df


def plot_sector_lines(sector_df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    """Create a line chart for one sentiment method."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=sector_df,
        x="quarter",
        y=column,
        hue="sector",
        marker="o",
        linewidth=2.2,
    )
    y_values = sector_df[column].dropna()
    if not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_range = y_max - y_min
        padding = max(y_range * 0.15, 0.01)
        plt.ylim(y_min - padding, y_max + padding)

        zero_threshold = padding * 0.5
        if y_min - zero_threshold <= 0 <= y_max + zero_threshold:
            plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    else:
        plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
    plt.xlim(1, 4)
    plt.title(title)
    plt.xlabel("Quarter")
    plt.ylabel("Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_comparison(sector_df: pd.DataFrame, output_path: Path) -> None:
    """Overlay both methods in one comparison figure."""
    plot_df = sector_df.melt(
        id_vars=["sector", "year", "quarter"],
        value_vars=["sentiment_lexicon", "sentiment_finbert"],
        var_name="method",
        value_name="sentiment",
    )
    plot_df["method"] = plot_df["method"].map(
        {
            "sentiment_lexicon": "Lexicon",
            "sentiment_finbert": "FinBERT",
        }
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=plot_df,
        x="quarter",
        y="sentiment",
        hue="sector",
        style="method",
        markers=True,
        dashes=True,
        linewidth=2,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.title("Sector Outlook Sentiment by Quarter: Lexicon vs FinBERT")
    plt.xlabel("Quarter")
    plt.ylabel("Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(firm_quarter_df: pd.DataFrame, output_path: Path) -> None:
    """Show firm-quarter sentiment distribution by sector."""
    plot_df = firm_quarter_df.melt(
        id_vars=["sector"],
        value_vars=["sentiment_lexicon", "sentiment_finbert"],
        var_name="method",
        value_name="sentiment",
    )
    plot_df["method"] = plot_df["method"].map(
        {
            "sentiment_lexicon": "Lexicon",
            "sentiment_finbert": "FinBERT",
        }
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="sector", y="sentiment", hue="method")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.title("Firm-Quarter Sentiment Distribution by Sector")
    plt.xlabel("Sector")
    plt.ylabel("Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar_chart(sector_df: pd.DataFrame, output_path: Path) -> None:
    """Plot average sentiment per sector."""
    bar_df = (
        sector_df.groupby("sector", as_index=False)[["sentiment_lexicon", "sentiment_finbert"]]
        .mean()
        .melt(id_vars="sector", var_name="method", value_name="sentiment")
    )
    bar_df["method"] = bar_df["method"].map(
        {
            "sentiment_lexicon": "Lexicon",
            "sentiment_finbert": "FinBERT",
        }
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(data=bar_df, x="sector", y="sentiment", hue="method")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.title("Average Outlook Sentiment by Sector")
    plt.xlabel("Sector")
    plt.ylabel("Average Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(firm_quarter_df: pd.DataFrame, sector: str, output_path: Path) -> None:
    """Create a company-quarter FinBERT heatmap for one sector."""
    subset = firm_quarter_df[firm_quarter_df["sector"] == sector].copy()
    if subset.empty:
        return

    subset["quarter_label"] = "Q" + subset["quarter"].astype(int).astype(str)
    pivot = subset.pivot_table(
        index="symbol",
        columns="quarter_label",
        values="sentiment_finbert",
        aggfunc="mean",
    )
    quarter_order = [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in pivot.columns]
    pivot = pivot.reindex(columns=quarter_order)

    plt.figure(figsize=(8, max(6, 0.35 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5)
    plt.title(f"{sector} Company-Quarter Sentiment Heatmap (FinBERT)")
    plt.xlabel("Quarter")
    plt.ylabel("Symbol")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(
    outlook_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    firm_quarter_df: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> None:
    """Save detailed and aggregated outputs."""
    outlook_output = outlook_df[
        [
            "company",
            "symbol",
            "sector",
            "year",
            "quarter",
            "date",
            "sentence",
            "sentiment_lexicon",
            "sentiment_finbert",
        ]
    ].copy()
    outlook_output.to_csv(OUTPUT_DIR / "outlook_sentences.csv", index=False, encoding="utf-8-sig")

    firm_quarter_df.to_csv(OUTPUT_DIR / "firm_quarter_sentiment.csv", index=False, encoding="utf-8-sig")
    sector_df.to_csv(OUTPUT_DIR / "sector_sentiment.csv", index=False, encoding="utf-8-sig")

    transcript_output = transcript_df[
        [
            "sector",
            "company",
            "symbol",
            "date",
            "year",
            "quarter",
            "outlook_sentence_count",
            "sentiment_lexicon",
            "sentiment_finbert",
            "outlook_text",
        ]
    ].copy()
    transcript_output.to_csv(OUTPUT_DIR / "transcript_sentiment_details.csv", index=False, encoding="utf-8-sig")

    sentiment_results = sector_df[
        ["sector", "year", "quarter", "sentiment_lexicon", "sentiment_finbert"]
    ].copy()
    sentiment_results.to_csv(OUTPUT_DIR / "sentiment_results.csv", index=False, encoding="utf-8-sig")


def print_summary_statistics(firm_quarter_df: pd.DataFrame, sector_df: pd.DataFrame) -> None:
    """Print summary stats requested for quick interpretation."""
    sector_means = (
        sector_df.groupby("sector")[["sentiment_lexicon", "sentiment_finbert"]]
        .mean()
        .round(4)
    )
    print("\nAverage sentiment by sector:")
    print(sector_means)

    if {"Tech", "Industrials"}.issubset(set(sector_df["sector"].unique())):
        tech_mean = sector_df[sector_df["sector"] == "Tech"][["sentiment_lexicon", "sentiment_finbert"]].mean()
        ind_mean = sector_df[sector_df["sector"] == "Industrials"][["sentiment_lexicon", "sentiment_finbert"]].mean()
        diff = (tech_mean - ind_mean).round(4)
        print("\nDifference between Tech and Industrials (Tech - Industrials):")
        print(diff)

    firm_rank = firm_quarter_df.copy()
    firm_rank["combined_sentiment"] = firm_rank[["sentiment_lexicon", "sentiment_finbert"]].mean(axis=1)
    firm_rank = firm_rank.dropna(subset=["combined_sentiment"])

    if not firm_rank.empty:
        most_positive = firm_rank.nlargest(1, "combined_sentiment").iloc[0]
        most_negative = firm_rank.nsmallest(1, "combined_sentiment").iloc[0]

        print("\nMost positive company-quarter:")
        print(
            f"{most_positive['company']} ({most_positive['symbol']}), "
            f"{most_positive['sector']} Q{int(most_positive['quarter'])} {int(most_positive['year'])}, "
            f"combined sentiment={most_positive['combined_sentiment']:.4f}"
        )

        print("\nMost negative company-quarter:")
        print(
            f"{most_negative['company']} ({most_negative['symbol']}), "
            f"{most_negative['sector']} Q{int(most_negative['quarter'])} {int(most_negative['year'])}, "
            f"combined sentiment={most_negative['combined_sentiment']:.4f}"
        )


def main() -> None:
    ensure_output_dir()
    sns.set_theme(style="whitegrid")

    print("Loading transcript CSV files...")
    df = load_and_combine_data()

    print("Splitting transcripts into sentences...")
    sentence_df = build_sentence_level_df(df)

    print("Filtering outlook sentences...")
    outlook_df = filter_outlook_sentences(sentence_df)
    if outlook_df.empty:
        raise SystemExit("No outlook sentences were found. Check the transcript inputs or keyword list.")

    print("Computing lexicon and FinBERT sentiment...")
    outlook_df = score_outlook_sentences(outlook_df)

    print("Aggregating to transcript, firm-quarter, and sector-quarter...")
    transcript_df, firm_quarter_df, sector_df = aggregate_sentiment(outlook_df)

    print("Saving CSV outputs...")
    save_outputs(outlook_df, transcript_df, firm_quarter_df, sector_df)

    print("Saving plots...")
    plot_sector_lines(
        sector_df,
        column="sentiment_lexicon",
        title="Sector Outlook Sentiment by Quarter: Lexicon Method",
        output_path=OUTPUT_DIR / "sector_sentiment_lexicon.png",
    )
    plot_sector_lines(
        sector_df,
        column="sentiment_finbert",
        title="Sector Outlook Sentiment by Quarter: FinBERT Method",
        output_path=OUTPUT_DIR / "sector_sentiment_finbert.png",
    )
    plot_combined_comparison(sector_df, OUTPUT_DIR / "sector_sentiment_combined.png")
    plot_boxplot(firm_quarter_df, OUTPUT_DIR / "firm_sentiment_boxplot.png")
    plot_bar_chart(sector_df, OUTPUT_DIR / "avg_sector_sentiment_bar.png")
    plot_heatmap(firm_quarter_df, "Tech", OUTPUT_DIR / "tech_company_quarter_heatmap_finbert.png")
    plot_heatmap(
        firm_quarter_df,
        "Industrials",
        OUTPUT_DIR / "industrials_company_quarter_heatmap_finbert.png",
    )

    print_summary_statistics(firm_quarter_df, sector_df)

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'outlook_sentences.csv'}")
    print(f"- {OUTPUT_DIR / 'firm_quarter_sentiment.csv'}")
    print(f"- {OUTPUT_DIR / 'sector_sentiment.csv'}")
    print(f"- {OUTPUT_DIR / 'sentiment_results.csv'}")
    print(f"- {OUTPUT_DIR / 'transcript_sentiment_details.csv'}")
    print(f"- {OUTPUT_DIR / 'sector_sentiment_lexicon.png'}")
    print(f"- {OUTPUT_DIR / 'sector_sentiment_finbert.png'}")
    print(f"- {OUTPUT_DIR / 'sector_sentiment_combined.png'}")
    print(f"- {OUTPUT_DIR / 'firm_sentiment_boxplot.png'}")
    print(f"- {OUTPUT_DIR / 'avg_sector_sentiment_bar.png'}")
    print(f"- {OUTPUT_DIR / 'tech_company_quarter_heatmap_finbert.png'}")
    print(f"- {OUTPUT_DIR / 'industrials_company_quarter_heatmap_finbert.png'}")


if __name__ == "__main__":
    main()
