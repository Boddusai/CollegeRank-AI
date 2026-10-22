from pathlib import Path
import pandas as pd
import re

RAW = Path("data/raw/jee_main")
OUT = Path("data/processed/jee_main_clean.csv")

ALIASES = {
    "college": ["college", "institute", "institute name", "Institute"],
    "branch": ["branch", "program", "academic program", "Academic Program Name"],
    "category": ["category", "seat type", "Seat Type"],
    "gender": ["gender", "Gender"],
    "quota": ["quota", "Quota"],
    "round": ["round", "Round"],
    "opening_rank": ["opening rank", "Opening Rank", "Opening Rank (Overall)"],
    "closing_rank": ["closing rank", "Closing Rank", "Closing Rank (Overall)"],
    "year": ["year", "Year"],
}


def normalize_columns(df):
    normalized = {}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for target, options in ALIASES.items():
        for option in options:
            key = option.strip().lower()
            if key in lower:
                normalized[target] = lower[key]
                break
    missing = [k for k in ALIASES if k not in normalized]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")
    return df.rename(columns={v: k for k, v in normalized.items()})


def clean_text(s):
    return (
        s.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/jee_main/. Add verified JoSAA historical CSV files first."
        )

    frames = []
    for file in files:
        df = pd.read_csv(file)
        df = normalize_columns(df)
        if df["year"].isna().all():
            m = re.search(r"(20\d{2})", file.name)
            if m:
                df["year"] = int(m.group(1))
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    for col in ["college", "branch", "category", "gender", "quota"]:
        df[col] = clean_text(df[col])

    for col in ["year", "round", "opening_rank", "closing_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "college", "branch", "closing_rank"])
    df = df[(df["year"] >= 2021) & (df["year"] <= 2025)]
    df = df[df["closing_rank"] > 0]
    df = df.drop_duplicates()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df):,} cleaned rows to {OUT}")
    print(df.groupby("year").size())


if __name__ == "__main__":
    main()
