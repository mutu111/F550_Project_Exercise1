"""
Download and prepare FY2024 earnings-call transcripts for:
1. Information Technology
2. Industrials

The script loads the Hugging Face dataset:
    kurry/sp500_earnings_transcripts

It then:
1. Downloads/reads the dataset
2. Filters to FY2024 only using dataset year/quarter labels
3. Filters to Information Technology and Industrials only
4. Keeps one transcript per company-quarter
5. Selects 20 companies per sector with full four-quarter coverage
6. Exports transcript-level CSV files for later analysis

Output files:
    outputs/tech_20_companies_transcripts_2024.csv
    outputs/industrials_20_companies_transcripts_2024.csv
    outputs/selected_companies_summary_2024.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    missing_pkg = exc.name or "datasets"
    raise SystemExit(
        "Missing dependency: "
        f"`{missing_pkg}`.\n\n"
        "Install with:\n"
        "  pip install -r requirements.txt"
    ) from exc


DATASET_NAME = "kurry/sp500_earnings_transcripts"
TARGET_FISCAL_YEAR = 2024
OUTPUT_DIR = Path("outputs")
MIN_TRANSCRIPTS_PER_COMPANY = 4
TARGET_COMPANIES_PER_SECTOR = 20

# We prefer a stable, presentation-friendly sample of large and recognizable names.
PREFERRED_TECH = [
    "AAPL",
    "MSFT",
    "NVDA",
    "ORCL",
    "CRM",
    "ADBE",
    "AMD",
    "QCOM",
    "CSCO",
    "IBM",
]

PREFERRED_INDUSTRIALS = [
    "CAT",
    "HON",
    "GE",
    "RTX",
    "BA",
    "UPS",
    "UNP",
    "DE",
    "ETN",
    "LMT",
]

SECTOR_MAP: Dict[str, str] = {
    # Information Technology
    "ACN": "Information Technology",
    "ADBE": "Information Technology",
    "ADI": "Information Technology",
    "ADP": "Information Technology",
    "AKAM": "Information Technology",
    "AMD": "Information Technology",
    "AMAT": "Information Technology",
    "ANET": "Information Technology",
    "ANSS": "Information Technology",
    "APH": "Information Technology",
    "AAPL": "Information Technology",
    "CDNS": "Information Technology",
    "CRM": "Information Technology",
    "CSCO": "Information Technology",
    "CTSH": "Information Technology",
    "DXC": "Information Technology",
    "ENPH": "Information Technology",
    "EPAM": "Information Technology",
    "FICO": "Information Technology",
    "FTNT": "Information Technology",
    "GEN": "Information Technology",
    "GLW": "Information Technology",
    "GPN": "Information Technology",
    "HPE": "Information Technology",
    "HPQ": "Information Technology",
    "IBM": "Information Technology",
    "INTC": "Information Technology",
    "INTU": "Information Technology",
    "IT": "Information Technology",
    "JBL": "Information Technology",
    "JNPR": "Information Technology",
    "KEYS": "Information Technology",
    "KLAC": "Information Technology",
    "LRCX": "Information Technology",
    "MA": "Information Technology",
    "MCHP": "Information Technology",
    "MPWR": "Information Technology",
    "MSFT": "Information Technology",
    "MSI": "Information Technology",
    "MU": "Information Technology",
    "NTAP": "Information Technology",
    "NVDA": "Information Technology",
    "NXPI": "Information Technology",
    "ON": "Information Technology",
    "ORCL": "Information Technology",
    "PANW": "Information Technology",
    "PAYX": "Information Technology",
    "PLTR": "Information Technology",
    "PTC": "Information Technology",
    "PYPL": "Information Technology",
    "QCOM": "Information Technology",
    "SNPS": "Information Technology",
    "TEL": "Information Technology",
    "TER": "Information Technology",
    "TRMB": "Information Technology",
    "TXN": "Information Technology",
    "TYL": "Information Technology",
    "V": "Information Technology",
    "VRSN": "Information Technology",
    "WDC": "Information Technology",
    "ZBRA": "Information Technology",
    # Industrials
    "AAL": "Industrials",
    "AOS": "Industrials",
    "BA": "Industrials",
    "BRO": "Industrials",
    "CARR": "Industrials",
    "CAT": "Industrials",
    "CHRW": "Industrials",
    "CMI": "Industrials",
    "CPRT": "Industrials",
    "CSX": "Industrials",
    "CTAS": "Industrials",
    "DAL": "Industrials",
    "DE": "Industrials",
    "DOV": "Industrials",
    "EFX": "Industrials",
    "EMR": "Industrials",
    "ETN": "Industrials",
    "EXPD": "Industrials",
    "FAST": "Industrials",
    "FDX": "Industrials",
    "GD": "Industrials",
    "GE": "Industrials",
    "GEV": "Industrials",
    "GNRC": "Industrials",
    "GWW": "Industrials",
    "HON": "Industrials",
    "HUBB": "Industrials",
    "HWM": "Industrials",
    "IR": "Industrials",
    "ITW": "Industrials",
    "J": "Industrials",
    "JBHT": "Industrials",
    "JCI": "Industrials",
    "LHX": "Industrials",
    "LMT": "Industrials",
    "MAS": "Industrials",
    "MMM": "Industrials",
    "NDSN": "Industrials",
    "NOC": "Industrials",
    "NSC": "Industrials",
    "ODFL": "Industrials",
    "OTIS": "Industrials",
    "PCAR": "Industrials",
    "PH": "Industrials",
    "PNR": "Industrials",
    "PWR": "Industrials",
    "ROK": "Industrials",
    "ROL": "Industrials",
    "RTX": "Industrials",
    "SNA": "Industrials",
    "SWK": "Industrials",
    "TDG": "Industrials",
    "TXT": "Industrials",
    "UAL": "Industrials",
    "UNP": "Industrials",
    "UPS": "Industrials",
    "URI": "Industrials",
    "VLTO": "Industrials",
    "WM": "Industrials",
    "XYL": "Industrials",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_dataset_year(record: dict) -> Optional[int]:
    """Get the dataset's year label, which we treat as the fiscal/reporting year."""
    year_value = record.get("year")
    if year_value is not None and str(year_value).strip():
        try:
            return int(year_value)
        except (TypeError, ValueError):
            return None

    return None


def parse_quarter(record: dict) -> Optional[int]:
    """Normalize quarter values into integers 1-4."""
    quarter_value = record.get("quarter")
    if quarter_value is None:
        return None

    quarter_text = str(quarter_value).strip().upper()
    if not quarter_text:
        return None

    quarter_text = quarter_text.replace("Q", "")
    try:
        quarter_num = int(float(quarter_text))
    except (TypeError, ValueError):
        return None

    if quarter_num in {1, 2, 3, 4}:
        return quarter_num
    return None


def normalize_sector(record: dict) -> Optional[str]:
    """Use the dataset sector if present; otherwise fall back to manual ticker mapping."""
    ticker = str(record.get("symbol", "")).upper().strip()
    raw_sector = record.get("sector")
    if raw_sector is not None and str(raw_sector).strip():
        sector = str(raw_sector).strip()
        if sector in {"Information Technology", "Industrials"}:
            return sector
    return SECTOR_MAP.get(ticker)


def load_filtered_transcripts_fy2024() -> pd.DataFrame:
    """
    Load the dataset in streaming mode and filter immediately.

    Filtering order:
    1. Keep only fiscal/report year 2024
    2. Keep only quarters 1-4
    3. Keep only Information Technology and Industrials
    4. Keep only the columns needed downstream
    """
    stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    rows = []

    for record in stream:
        year = parse_dataset_year(record)
        quarter = parse_quarter(record)

        if year != TARGET_FISCAL_YEAR:
            continue
        if quarter not in {1, 2, 3, 4}:
            continue

        sector = normalize_sector(record)
        if sector not in {"Information Technology", "Industrials"}:
            continue

        rows.append(
            {
                "symbol": str(record.get("symbol", "")).upper().strip(),
                "company_name": str(record.get("company_name", "")).strip(),
                "date": record.get("date"),
                "year": year,
                "quarter": quarter,
                "content": str(record.get("content", "") or ""),
                "sector": sector,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["symbol", "company_name", "date", "year", "quarter", "content", "sector"]
        )

    return pd.DataFrame(rows)


def keep_one_transcript_per_company_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one transcript per company-quarter.

    If duplicates exist for the same company and quarter, retain the latest dated transcript.
    """
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper().str.strip()
    work["company_name"] = work["company_name"].astype(str).str.strip()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["content"] = work["content"].fillna("").astype(str)
    work = work.sort_values(["sector", "symbol", "year", "quarter", "date"]).copy()
    work = work.drop_duplicates(
        subset=["sector", "symbol", "year", "quarter"],
        keep="last",
    ).reset_index(drop=True)
    return work


def choose_companies(
    df: pd.DataFrame,
    sector: str,
    preferred_list: List[str],
    n_companies: int = TARGET_COMPANIES_PER_SECTOR,
    min_transcripts: int = MIN_TRANSCRIPTS_PER_COMPANY,
) -> List[str]:
    """
    Select up to n_companies for a sector.

    Priority:
    1. Keep only companies with all four FY2024 quarters represented
    2. Keep only companies with at least `min_transcripts`
    3. Use preferred company list first
    4. Fill remaining slots with other companies ranked by transcript count
    """
    sector_df = df[df["sector"] == sector].copy()
    company_stats = (
        sector_df.groupby(["symbol", "company_name"], as_index=False)
        .agg(
            transcript_count=("content", "size"),
            quarter_count=("quarter", "nunique"),
        )
    )
    eligible = company_stats[
        (company_stats["transcript_count"] >= min_transcripts)
        & (company_stats["quarter_count"] == 4)
    ].copy()
    eligible = eligible.sort_values(
        ["quarter_count", "transcript_count", "symbol"],
        ascending=[False, False, True],
    )

    available = set(eligible["symbol"].dropna().unique())

    selected = [ticker for ticker in preferred_list if ticker in available]

    if len(selected) < n_companies:
        for ticker in eligible["symbol"]:
            if ticker not in selected:
                selected.append(ticker)
            if len(selected) == n_companies:
                break

    return selected[:n_companies]


def export_sector_csv(
    df: pd.DataFrame,
    sector: str,
    selected_tickers: List[str],
    output_name: str,
) -> pd.DataFrame:
    """Export transcript-level CSV for the chosen companies."""
    out_df = df[(df["sector"] == sector) & (df["symbol"].isin(selected_tickers))].copy()
    out_df = out_df.rename(columns={"company_name": "company", "content": "transcript"})
    out_df = out_df[
        ["company", "symbol", "sector", "date", "quarter", "year", "transcript"]
    ].sort_values(["symbol", "date"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_DIR / output_name, index=False, encoding="utf-8-sig")
    return out_df


def build_summary(tech_df: pd.DataFrame, industrial_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary of the selected companies and transcript counts."""
    combined = pd.concat([tech_df, industrial_df], ignore_index=True)
    summary = (
        combined.groupby(["sector", "symbol", "company"], as_index=False)
        .agg(
            transcript_count=("transcript", "size"),
            quarter_count=("quarter", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["sector", "symbol"])
        .reset_index(drop=True)
    )
    if not summary.empty:
        quarters_present = (
            combined.groupby(["sector", "symbol", "company"])["quarter"]
            .apply(lambda s: ",".join(f"Q{int(q)}" for q in sorted(pd.unique(s))))
            .reset_index(name="quarters_present")
        )
        summary = summary.merge(
            quarters_present,
            on=["sector", "symbol", "company"],
            how="left",
        )
    summary.to_csv(OUTPUT_DIR / "selected_companies_summary_2024.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    ensure_output_dir()

    print(f"Loading dataset from Hugging Face and filtering to FY{TARGET_FISCAL_YEAR}...")
    raw_df = load_filtered_transcripts_fy2024()

    print("Keeping only Information Technology and Industrials transcripts...")
    data_2024 = keep_one_transcript_per_company_quarter(raw_df)

    tech_tickers = choose_companies(
        data_2024,
        sector="Information Technology",
        preferred_list=PREFERRED_TECH,
        n_companies=TARGET_COMPANIES_PER_SECTOR,
        min_transcripts=MIN_TRANSCRIPTS_PER_COMPANY,
    )
    industrial_tickers = choose_companies(
        data_2024,
        sector="Industrials",
        preferred_list=PREFERRED_INDUSTRIALS,
        n_companies=TARGET_COMPANIES_PER_SECTOR,
        min_transcripts=MIN_TRANSCRIPTS_PER_COMPANY,
    )

    print("Selected Information Technology tickers:", tech_tickers)
    print("Selected Industrials tickers:", industrial_tickers)
    if len(tech_tickers) < TARGET_COMPANIES_PER_SECTOR:
        print(
            f"Warning: only found {len(tech_tickers)} Information Technology companies "
            f"with full FY{TARGET_FISCAL_YEAR} four-quarter transcript coverage."
        )
    if len(industrial_tickers) < TARGET_COMPANIES_PER_SECTOR:
        print(
            f"Warning: only found {len(industrial_tickers)} Industrials companies "
            f"with full FY{TARGET_FISCAL_YEAR} four-quarter transcript coverage."
        )

    tech_df = export_sector_csv(
        data_2024,
        sector="Information Technology",
        selected_tickers=tech_tickers,
        output_name="tech_20_companies_transcripts_2024.csv",
    )
    industrial_df = export_sector_csv(
        data_2024,
        sector="Industrials",
        selected_tickers=industrial_tickers,
        output_name="industrials_20_companies_transcripts_2024.csv",
    )

    summary_df = build_summary(tech_df, industrial_df)

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'tech_20_companies_transcripts_2024.csv'}")
    print(f"- {OUTPUT_DIR / 'industrials_20_companies_transcripts_2024.csv'}")
    print(f"- {OUTPUT_DIR / 'selected_companies_summary_2024.csv'}")

    print("\nTranscript counts by sector:")
    print(summary_df.groupby("sector")["transcript_count"].sum())


if __name__ == "__main__":
    main()
