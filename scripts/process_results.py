import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_WIDTH = 0.05

DEFAULT_FILES = [
    "../results/DeepPrint/fvc_2000_db1_a.csv",
    "../results/DeepPrint/fvc_2000_db1_b.csv",
    "../results/DeepPrint/fvc_2000_db2_a.csv",
    "../results/DeepPrint/fvc_2000_db2_b.csv",
    "../results/DeepPrint/fvc_2000_db3_a.csv",
    "../results/DeepPrint/fvc_2000_db3_b.csv",
    "../results/DeepPrint/fvc_2000_db4_a.csv",
    "../results/DeepPrint/fvc_2000_db4_b.csv",
    "../results/DeepPrint/fvc_2002_db1_a.csv",
    "../results/DeepPrint/fvc_2002_db1_b.csv",
    "../results/DeepPrint/fvc_2002_db2_a.csv",
    "../results/DeepPrint/fvc_2002_db2_b.csv",
    "../results/DeepPrint/fvc_2002_db3_a.csv",
    "../results/DeepPrint/fvc_2002_db3_b.csv",
    "../results/DeepPrint/fvc_2002_db4_a.csv",
    "../results/DeepPrint/fvc_2002_db4_b.csv",
    "../results/DeepPrint/fvc_2004_db1_a.csv",
    "../results/DeepPrint/fvc_2004_db1_b.csv",
    "../results/DeepPrint/fvc_2004_db2_a.csv",
    "../results/DeepPrint/fvc_2004_db2_b.csv",
    "../results/DeepPrint/fvc_2004_db3_a.csv",
    "../results/DeepPrint/fvc_2004_db3_b.csv",
    "../results/DeepPrint/fvc_2004_db4_a.csv",
    "../results/DeepPrint/fvc_2004_db4_b.csv",
    "../results/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
]


def derive_metrics_csv_path(input_csv: str | Path) -> Path:
    csv_path = Path(input_csv)
    if csv_path.name.endswith("_metrics.csv"):
        return csv_path
    if csv_path.name.endswith(".csv"):
        return csv_path.with_name(csv_path.name[:-4] + "_metrics.csv")
    return csv_path.with_name(csv_path.name + "_metrics.csv")


def calculate_metrics(input_csv: str | Path, output_csv: str | Path | None = None) -> None:
    csv_path = (SCRIPT_DIR / input_csv).resolve()
    if not csv_path.is_file() or not csv_path.name.endswith(".csv"):
        print(f"Skipping invalid input csv: {input_csv}")
        return

    if output_csv is None:
        out_path = derive_metrics_csv_path(csv_path)
    else:
        out_path = (SCRIPT_DIR / output_csv).resolve()

    comps = pd.read_csv(csv_path)

    min_score = comps["score"].min()
    max_score = comps["score"].max()
    comps["score"] = (comps["score"] - min_score) / (max_score - min_score) * 2.0

    thresholds = comps["score"].unique()
    min_t, max_t = thresholds.max(), thresholds.min()
    sample_thresholds = np.linspace(min_t, max_t, 1000)

    results_series = [None] * len(sample_thresholds)

    for i, t in enumerate(sample_thresholds):
        users = comps["user_1"].unique()
        frr_per_user = [0.0] * len(users)
        far_per_user = [0.0] * len(users)
        for j, user in enumerate(users):
            genuine_attempts = comps[(comps["user_1"] == user) & (comps["user_2"] == user)]
            impostor_attempts = comps[(comps["user_1"] == user) & (comps["user_2"] != user)]

            true_positives = (genuine_attempts["score"] >= t).sum()
            false_rejections = (genuine_attempts["score"] < t).sum()
            true_negatives = (impostor_attempts["score"] < t).sum()
            false_acceptances = (impostor_attempts["score"] >= t).sum()

            total_genuine = false_rejections + true_positives
            total_impostor = false_acceptances + true_negatives

            frr_per_user[j] = false_rejections / total_genuine if total_genuine > 0 else 0.0
            far_per_user[j] = false_acceptances / total_impostor if total_impostor > 0 else 0.0

        results_series[i] = pd.Series({
            "frr": np.mean(frr_per_user),
            "far": np.mean(far_per_user),
            "threshold": t,
        })

    result = pd.DataFrame(results_series)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Metrics saved to: {out_path}")


def plot_histogram(input_csv: str | Path, bin_width: float = BIN_WIDTH) -> None:
    csv_path = (SCRIPT_DIR / input_csv).resolve()
    if not csv_path.is_file():
        print(f"File not found: {input_csv}")
        return

    df = pd.read_csv(csv_path)

    genuine = df[df["user_1"] == df["user_2"]]["score"]
    impostor = df[df["user_1"] != df["user_2"]]["score"]

    all_scores = pd.concat([genuine, impostor])
    bin_edges = np.arange(
        np.floor(all_scores.min() / bin_width) * bin_width,
        np.ceil(all_scores.max() / bin_width) * bin_width + bin_width,
        bin_width,
    )

    plt.figure(figsize=(10, 6))
    plt.hist(genuine, bins=bin_edges, alpha=0.5, label="Mesmo usuário", color="blue", edgecolor="black")
    plt.hist(impostor, bins=bin_edges, alpha=0.5, label="Usuários diferentes", color="red", edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Quantidade")
    plt.title(csv_path.stem)
    plt.legend()
    plt.tight_layout()

    output_png = csv_path.with_suffix(".png")
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Histogram saved to: {output_png}")


def find_min_diff(csv_path: str | Path) -> dict | None:
    path = (SCRIPT_DIR / csv_path).resolve()
    if not path.is_file():
        print(f"File not found: {csv_path}")
        return None

    best_row = None
    best_diff = float("inf")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frr = float(row["frr"])
            far = float(row["far"])
            diff = abs(frr - far)
            if diff < best_diff:
                best_diff = diff
                best_row = row

    if best_row is None:
        print(f"No data found in {csv_path}")
        return None

    frr = round(float(best_row["frr"]), 3)
    far = round(float(best_row["far"]), 3)
    err = round((float(best_row["frr"]) + float(best_row["far"])) / 2.0, 3)
    threshold = round(float(best_row["threshold"]), 3)

    path_parts = [p.lower() for p in path.parts]
    if "deepprint" in path_parts:
        stem = path.name[:-12] if path.name.endswith("_metrics.csv") else (path.name[:-4] if path.name.endswith(".csv") else path.name)
        filename = f"deep_print_{stem}"
    elif "flare" in path_parts:
        idx = path_parts.index("flare")
        db_name = path_parts[idx + 1]
        filename = f"flare_{db_name}"
    else:
        filename = path.name
        if filename.endswith("_metrics.csv"):
            filename = filename[:-12]
        elif filename.endswith(".csv"):
            filename = filename[:-4]

    print(f"File: {path.name}")
    print(f"Row with minimal |frr - far| (diff = {best_diff:.10f}):")
    print(f"  far = {far}")
    print(f"  frr = {frr}")
    print(f"  err = {err}")
    print(f"  threshold = {threshold}")
    return {
        "file": filename,
        "far": far,
        "frr": frr,
        "err": err,
        "threshold": threshold,
    }


def run_metrics_command(files: list[str]) -> None:
    for input_csv in files:
        print(f"Calculating metrics for: {input_csv}")
        calculate_metrics(input_csv)


def run_histogram_command(files: list[str], bin_width: float) -> None:
    for input_csv in files:
        print(f"Plotting histogram for: {input_csv}")
        plot_histogram(input_csv, bin_width)


def run_min_diff_command(files: list[str], output_csv: str | Path = "../results/min_diff.csv") -> None:
    results = []
    for input_csv in files:
        metrics_csv = derive_metrics_csv_path(input_csv)
        print(f"Finding min diff for: {metrics_csv}")
        row_dict = find_min_diff(metrics_csv)
        if row_dict is not None:
            results.append(row_dict)

    if results:
        out_path = (SCRIPT_DIR / output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_dataframe = pd.DataFrame(results, columns=["file", "far", "frr", "err", "threshold"])
        results_dataframe.to_csv(out_path, index=False)
        print(f"Min diff results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biometric score analysis tool combining metrics calculation, histogram plotting, and minimum difference analysis.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    metrics_parser = subparsers.add_parser("metrics", help="Calculate FRR and FAR metrics across thresholds.")
    metrics_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score CSV files to process.")

    histogram_parser = subparsers.add_parser("histogram", help="Plot score distribution histograms.")
    histogram_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score CSV files to process.")
    histogram_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")

    min_diff_parser = subparsers.add_parser("min-diff", help="Find row with minimal |frr - far| from metrics CSV files.")
    min_diff_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score CSV files (or metrics CSV files) to process.")
    min_diff_parser.add_argument("--output", default="../results/min_diff.csv", help="Output CSV file path for min diff results.")

    all_parser = subparsers.add_parser("all", help="Run metrics, histogram, and min-diff analysis on target files.")
    all_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score CSV files to process.")
    all_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")
    all_parser.add_argument("--output", default="../results/min_diff.csv", help="Output CSV file path for min diff results.")

    args = parser.parse_args()

    if args.command == "metrics":
        run_metrics_command(args.files)
    elif args.command == "histogram":
        run_histogram_command(args.files, args.bin_width)
    elif args.command == "min-diff":
        run_min_diff_command(args.files, args.output)
    elif args.command == "all":
        run_metrics_command(args.files)
        run_histogram_command(args.files, args.bin_width)
        run_min_diff_command(args.files, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

