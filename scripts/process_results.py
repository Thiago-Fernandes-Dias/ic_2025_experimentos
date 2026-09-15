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
    "../results/DeepPrint_priorenh/fvc_2000_db1_a.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db1_b.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db2_a.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db2_b.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db3_a.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db3_b.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db4_a.csv",
    "../results/DeepPrint_priorenh/fvc_2000_db4_b.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db1_a.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db1_b.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db2_a.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db2_b.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db3_a.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db3_b.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db4_a.csv",
    "../results/DeepPrint_priorenh/fvc_2002_db4_b.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db1_a.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db1_b.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db2_a.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db2_b.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db3_a.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db3_b.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db4_a.csv",
    "../results/DeepPrint_priorenh/fvc_2004_db4_b.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db1_a.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db1_b.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db2_a.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db2_b.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db3_a.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db3_b.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db4_a.csv",
    "../results/DeepPrint_unetenh/fvc_2000_db4_b.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db1_a.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db1_b.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db2_a.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db2_b.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db3_a.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db3_b.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db4_a.csv",
    "../results/DeepPrint_unetenh/fvc_2002_db4_b.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db1_a.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db1_b.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db2_a.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db2_b.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db3_a.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db3_b.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db4_a.csv",
    "../results/DeepPrint_unetenh/fvc_2004_db4_b.csv",
    "../results/FLARE/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_PRIORENH/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv",
    "../results/FLARE_UNETENH/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv"
]


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{suffix}'. Expected .parquet or .csv")


def save_dataframe(dataframe: pd.DataFrame, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataframe.to_parquet(path, index=False)
    elif suffix == ".csv":
        dataframe.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Expected .parquet or .csv")


def resolve_existing_file(file_path: str | Path) -> Path:
    path = (SCRIPT_DIR / file_path).resolve()
    if path.is_file():
        return path
    alt_suffix = ".parquet" if path.suffix.lower() == ".csv" else ".csv"
    alt_path = path.with_suffix(alt_suffix)
    if alt_path.is_file():
        return alt_path
    return path


def derive_metrics_path(input_path: str | Path, output_format: str | None = None) -> Path:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if output_format is not None:
        clean_format = output_format.lstrip(".").lower()
        target_suffix = f".{clean_format}"
    elif suffix in (".parquet", ".csv"):
        target_suffix = suffix
    else:
        target_suffix = ".csv"

    stem = path.stem
    if stem.endswith("_metrics"):
        return path.with_suffix(target_suffix)

    return path.with_name(f"{stem}_metrics{target_suffix}")


derive_metrics_csv_path = derive_metrics_path


def calculate_metrics(
    input_file: str | Path,
    output_file: str | Path | None = None,
    output_format: str | None = None,
) -> None:
    resolved_path = resolve_existing_file(input_file)
    if not resolved_path.is_file() or resolved_path.suffix.lower() not in (".csv", ".parquet"):
        print(f"Skipping invalid input file: {input_file}")
        return

    if output_file is None:
        out_path = derive_metrics_path(resolved_path, output_format=output_format)
    else:
        out_path = (SCRIPT_DIR / output_file).resolve()

    comps = load_dataframe(resolved_path)

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
    save_dataframe(result, out_path)
    print(f"Metrics saved to: {out_path}")


def plot_histogram(input_file: str | Path, bin_width: float = BIN_WIDTH) -> None:
    resolved_path = resolve_existing_file(input_file)
    if not resolved_path.is_file() or resolved_path.suffix.lower() not in (".csv", ".parquet"):
        print(f"File not found: {input_file}")
        return

    df = load_dataframe(resolved_path)

    genuine = df[df["user_1"] == df["user_2"]]["score"]
    impostor = df[df["user_1"] != df["user_2"]]["score"]

    all_scores = pd.concat([genuine, impostor])
    bin_edges = np.arange(
        np.floor(all_scores.min() / bin_width) * bin_width,
        np.ceil(all_scores.max() / bin_width) * bin_width + bin_width,
        bin_width,
    )

    plt.figure(figsize=(10, 6))
    plt.hist(genuine, bins=bin_edges, alpha=0.05, label="Mesmo usuário", color="blue", edgecolor="black")
    plt.hist(impostor, bins=bin_edges, alpha=0.05, label="Usuários diferentes", color="red", edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Quantidade")
    plt.title(resolved_path.stem)
    plt.legend()
    plt.tight_layout()

    output_png = resolved_path.with_suffix(".png")
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Histogram saved to: {output_png}")


def find_min_diff(metrics_file: str | Path) -> dict | None:
    resolved_path = resolve_existing_file(metrics_file)
    if not resolved_path.is_file() or resolved_path.suffix.lower() not in (".csv", ".parquet"):
        print(f"File not found: {metrics_file}")
        return None

    df = load_dataframe(resolved_path)
    if df.empty:
        print(f"No data found in {metrics_file}")
        return None

    diff_series = (df["frr"].astype(float) - df["far"].astype(float)).abs()
    best_idx = diff_series.idxmin()
    best_row = df.loc[best_idx]
    best_diff = float(diff_series.loc[best_idx])

    frr = round(float(best_row["frr"]), 3)
    far = round(float(best_row["far"]), 3)
    err = round((frr + far) / 2.0, 3)
    threshold = round(float(best_row["threshold"]), 3)

    path_parts = [p.lower() for p in resolved_path.parts]
    stem = resolved_path.stem
    if stem.endswith("_metrics"):
        stem = stem[:-8]

    if "deepprint" in path_parts:
        filename = f"deep_print_{stem}"
    elif "flare" in path_parts:
        idx = path_parts.index("flare")
        db_name = path_parts[idx + 1]
        filename = f"flare_{db_name}"
    else:
        filename = stem

    print(f"File: {resolved_path.name}")
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


def run_metrics_command(files: list[str], output_format: str | None = None) -> None:
    for input_file in files:
        print(f"Calculating metrics for: {input_file}")
        calculate_metrics(input_file, output_format=output_format)


def run_histogram_command(files: list[str], bin_width: float) -> None:
    for input_file in files:
        print(f"Plotting histogram for: {input_file}")
        plot_histogram(input_file, bin_width)


def run_min_diff_command(files: list[str], output_file: str | Path = "../results/min_diff.csv") -> None:
    results = []
    for input_file in files:
        metrics_path = derive_metrics_path(input_file)
        resolved_metrics_path = resolve_existing_file(metrics_path)
        print(f"Finding min diff for: {resolved_metrics_path.name}")
        row_dict = find_min_diff(resolved_metrics_path)
        if row_dict is not None:
            results.append(row_dict)

    if results:
        out_path = (SCRIPT_DIR / output_file).resolve()
        results_dataframe = pd.DataFrame(results, columns=["file", "far", "frr", "err", "threshold"])
        save_dataframe(results_dataframe, out_path)
        print(f"Min diff results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biometric score analysis tool combining metrics calculation, histogram plotting, and minimum difference analysis.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    metrics_parser = subparsers.add_parser("metrics", help="Calculate FRR and FAR metrics across thresholds.")
    metrics_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    metrics_parser.add_argument("--format", choices=["csv", "parquet"], default=None, help="Output format for metrics files (defaults to input file format).")

    histogram_parser = subparsers.add_parser("histogram", help="Plot score distribution histograms.")
    histogram_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    histogram_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")

    min_diff_parser = subparsers.add_parser("min-diff", help="Find row with minimal |frr - far| from metrics files.")
    min_diff_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files or metrics files (.csv or .parquet) to process.")
    min_diff_parser.add_argument("--output", default="../results/min_diff.csv", help="Output file path for min diff results (.csv or .parquet).")

    all_parser = subparsers.add_parser("all", help="Run metrics, histogram, and min-diff analysis on target files.")
    all_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    all_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")
    all_parser.add_argument("--format", choices=["csv", "parquet"], default=None, help="Output format for metrics files (defaults to input file format).")
    all_parser.add_argument("--output", default="../results/min_diff.csv", help="Output file path for min diff results (.csv or .parquet).")

    args = parser.parse_args()

    if args.command == "metrics":
        run_metrics_command(args.files, args.format)
    elif args.command == "histogram":
        run_histogram_command(args.files, args.bin_width)
    elif args.command == "min-diff":
        run_min_diff_command(args.files, args.output)
    elif args.command == "all":
        run_metrics_command(args.files, args.format)
        run_histogram_command(args.files, args.bin_width)
        run_min_diff_command(args.files, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

