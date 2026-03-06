import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


EVAL_SETS = ["test", "private_test"]
LABEL_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
LABEL_MAPPING = {label: idx for idx, label in enumerate(LABEL_ORDER)}
VALID_INT_LABELS = set(range(len(LABEL_ORDER)))


def _extract_single_column(df: pd.DataFrame, file_path: Path) -> pd.Series:
    """Return the unique column of a CSV file as a Series."""
    if df.shape[1] != 1:
        raise ValueError(
            f"Expected exactly 1 column in {file_path}, got {df.shape[1]} columns: "
            f"{list(df.columns)}"
        )
    return df.iloc[:, 0]


def _normalize_labels(series: pd.Series, file_path: Path) -> pd.Series:
    """Convert labels to ordered integer classes for QWK computation.

    Accepted formats:
    - string labels in {"A", ..., "G"}
    - integer labels in {0, ..., 6}
    """
    if series.isna().any():
        raise ValueError(f"Found NaN values in {file_path}")

    # Case 1: labels are strings such as A, B, ..., G
    s_str = series.astype(str).str.strip().str.upper()
    if s_str.isin(LABEL_MAPPING).all():
        return s_str.map(LABEL_MAPPING).astype(int)

    # Case 2: labels are already integer-encoded
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.isna().any():
        invalid_values = sorted(pd.Series(series.astype(str)).unique().tolist())
        raise ValueError(
            f"Unsupported labels in {file_path}. Expected labels in {LABEL_ORDER} "
            f"or integer classes in {sorted(VALID_INT_LABELS)}. "
            f"Got values like: {invalid_values[:10]}"
        )

    s_num = s_num.astype(int)
    invalid_ints = sorted(set(s_num.unique()) - VALID_INT_LABELS)
    if invalid_ints:
        raise ValueError(
            f"Invalid integer labels in {file_path}: {invalid_ints}. "
            f"Expected integer classes in {sorted(VALID_INT_LABELS)}."
        )

    return s_num


def compute_qwk(
    predictions: pd.Series,
    targets: pd.Series,
    pred_path: Path,
    target_path: Path,
) -> float:
    """Compute the Quadratic Weighted Kappa."""
    y_pred = _normalize_labels(predictions, pred_path)
    y_true = _normalize_labels(targets, target_path)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Prediction length mismatch for {pred_path.name}: "
            f"{len(y_pred)} predictions vs {len(y_true)} targets"
        )

    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def main(reference_dir: Path, prediction_dir: Path, output_dir: Path) -> None:
    """Scoring program for the ADEME DPE Codabench challenge."""
    scores = {}

    for eval_set in EVAL_SETS:
        print(f"Scoring {eval_set}")

        pred_path = prediction_dir / f"{eval_set}_predictions.csv"
        target_path = reference_dir / f"{eval_set}_labels.csv"

        predictions_df = pd.read_csv(pred_path)
        targets_df = pd.read_csv(target_path)

        predictions = _extract_single_column(predictions_df, pred_path)
        targets = _extract_single_column(targets_df, target_path)

        scores[eval_set] = compute_qwk(
            predictions=predictions,
            targets=targets,
            pred_path=pred_path,
            target_path=target_path,
        )

    metadata_path = prediction_dir / "metadata.json"
    if metadata_path.exists():
        durations = json.loads(metadata_path.read_text())
        scores.update(**durations)
    else:
        print("metadata.json not found. Timing information will be omitted.")

    print(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for the ADEME DPE Codabench challenge"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="Directory containing the ground-truth labels",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="Directory containing the participant predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="Directory where scores.json will be written",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir),
    )