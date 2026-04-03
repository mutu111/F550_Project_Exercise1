# F500 project exercise 1

This project downloads FY2024 S&P 500 earnings call transcripts for the Information Technology and Industrials sectors, then measures forward-looking sentiment using both the Loughran-McDonald financial lexicon and FinBERT.

## Project structure

```text
F550/agent/
|-- data/
|   |-- Loughran-McDonald_MasterDictionary_1993-2025.csv
|   `-- LoughranMcDonald_MasterDictionary_2024.csv
|-- outputs
|-- data_preprocess.py
|-- environment.yml
|-- main.py
|-- pyproject.toml
|-- README.md
|-- requirements.txt
`-- .gitignore
```


## Setup


```bash
conda env create -f environment.yml
conda activate f500-exercise1
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
