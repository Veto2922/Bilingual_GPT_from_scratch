"""Data cleaning pipeline for pretraining splits."""

from pathlib import Path
import pandas as pd

from src.data_processing.data_cleaning import clean_data_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUT_DIR = INPUT_DIR / "cleaned"

INPUT_PATHS: list[Path] = [
    INPUT_DIR / "phase1_train_filtered.parquet",
    INPUT_DIR / "phase1_eval_filtered.parquet",
    INPUT_DIR / "phase2_train_filtered.parquet",
    INPUT_DIR / "phase2_eval_filtered.parquet",
]


def clean_phase(phase: str, input_path: Path, output_path: Path) -> None:
    """
    Clean text data for a specific dataset split and save the result.

    Args:
        phase: Identifier name for the dataset split (e.g., 'phase1_train').
        input_path: Path to the input parquet file.
        output_path: Path to write the cleaned parquet file.
    """
    print(f"==== clean_phase: {phase} ====")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    print(f"  Source        : {input_path}")
    print(f"  Rows          : {len(df):,}")

    cleaned_df = clean_data_frame(df)

    removed_rows = len(df) - len(cleaned_df)
    print(f"  Rows kept     : {len(cleaned_df):,}  (dropped {removed_rows:,} empty/short rows)")

    cleaned_df.to_parquet(output_path, index=False)
    print(f"  Saved to      : {output_path}")


def pipeline() -> None:
    """Run data cleaning pipeline on all configured input datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in INPUT_PATHS:
        split_name = "_".join(path.stem.split("_")[:2])
        output_path = OUTPUT_DIR / f"{split_name}_clean.parquet"

        clean_phase(phase=split_name, input_path=path, output_path=output_path)


if __name__ == "__main__":
    pipeline()