# F500 project exercise 1

This project downloads FY2024 S&P 500 earnings call transcripts for the Information Technology and Industrials sectors, then measures forward-looking sentiment using both the Loughran-McDonald financial lexicon and FinBERT.

The workflow is split into two steps:

1. `data_preprocess.py` downloads and filters transcript data from the Hugging Face dataset `kurry/sp500_earnings_transcripts`, then exports one CSV per sector.
2. `main.py` extracts outlook-related sentences, scores them with a lexicon method and FinBERT, and saves analysis tables plus plots.

## Project structure

```text
F550/agent/
|-- data/
|   |-- Loughran-McDonald_MasterDictionary_1993-2025.csv
|   `-- LoughranMcDonald_MasterDictionary_2024.csv
|-- outputs/
|   |-- industrials_20_companies_transcripts_2024.csv
|   |-- tech_20_companies_transcripts_2024.csv
|   |-- firm_quarter_sentiment.csv
|   |-- sector_sentiment.csv
|   |-- transcript_sentiment_details.csv
|   `-- ... other generated csv/png outputs
|-- data_preprocess.py
|-- environment.yml
|-- main.py
|-- pyproject.toml
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Datasets

The two Loughran-McDonald dictionary files are stored in the [`data/`](./data) folder:

- `data/Loughran-McDonald_MasterDictionary_1993-2025.csv`
- `data/LoughranMcDonald_MasterDictionary_2024.csv`

The analysis script currently uses `data/LoughranMcDonald_MasterDictionary_2024.csv` by default.

## Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate f500-exercise1
```

### Option 2: pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Generate the sector transcript CSVs:

```bash
python data_preprocess.py
```

Run the sentiment pipeline:

```bash
python main.py
```

All generated CSV files and figures are saved to `outputs/`.

## Notes

- The transcript source is the Hugging Face dataset `kurry/sp500_earnings_transcripts`.
- The project focuses on FY2024 transcripts for 20 companies per sector with full quarter coverage.
- Sentiment is measured using both a dictionary-based method and the `ProsusAI/finbert` model.
