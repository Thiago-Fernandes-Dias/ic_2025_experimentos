from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (SCRIPT_DIR / "../results").resolve()
METRICS_DIR = RESULTS_DIR / "metrics"
HISTOGRAMS_DIR = RESULTS_DIR / "histograms"

BIN_WIDTH = 0.005
DECIMAL_PLACES = 8
THRESHOLD_GROUP_SIZE = 200

HISTOGRAM_GENUINE_COLOR: tuple[float, float, float] = (0.15, 0.85, 0.35)
HISTOGRAM_IMPOSTOR_COLOR: tuple[float, float, float] = (1.0, 0.25, 0.25)
HISTOGRAM_ALPHA = 0.6


DEFAULT_FILES: tuple[tuple[str, str], ...] = (
    ("../scores/DeepPrint/fvc_2000_db1_a.csv", "deep_print_fvc_2000_db1_a"),
    ("../scores/DeepPrint/fvc_2000_db1_b.csv", "deep_print_fvc_2000_db1_b"),
    ("../scores/DeepPrint/fvc_2000_db2_a.csv", "deep_print_fvc_2000_db2_a"),
    ("../scores/DeepPrint/fvc_2000_db2_b.csv", "deep_print_fvc_2000_db2_b"),
    ("../scores/DeepPrint/fvc_2000_db3_a.csv", "deep_print_fvc_2000_db3_a"),
    ("../scores/DeepPrint/fvc_2000_db3_b.csv", "deep_print_fvc_2000_db3_b"),
    ("../scores/DeepPrint/fvc_2000_db4_a.csv", "deep_print_fvc_2000_db4_a"),
    ("../scores/DeepPrint/fvc_2000_db4_b.csv", "deep_print_fvc_2000_db4_b"),
    ("../scores/DeepPrint/fvc_2002_db1_a.csv", "deep_print_fvc_2002_db1_a"),
    ("../scores/DeepPrint/fvc_2002_db1_b.csv", "deep_print_fvc_2002_db1_b"),
    ("../scores/DeepPrint/fvc_2002_db2_a.csv", "deep_print_fvc_2002_db2_a"),
    ("../scores/DeepPrint/fvc_2002_db2_b.csv", "deep_print_fvc_2002_db2_b"),
    ("../scores/DeepPrint/fvc_2002_db3_a.csv", "deep_print_fvc_2002_db3_a"),
    ("../scores/DeepPrint/fvc_2002_db3_b.csv", "deep_print_fvc_2002_db3_b"),
    ("../scores/DeepPrint/fvc_2002_db4_a.csv", "deep_print_fvc_2002_db4_a"),
    ("../scores/DeepPrint/fvc_2002_db4_b.csv", "deep_print_fvc_2002_db4_b"),
    ("../scores/DeepPrint/fvc_2004_db1_a.csv", "deep_print_fvc_2004_db1_a"),
    ("../scores/DeepPrint/fvc_2004_db1_b.csv", "deep_print_fvc_2004_db1_b"),
    ("../scores/DeepPrint/fvc_2004_db2_a.csv", "deep_print_fvc_2004_db2_a"),
    ("../scores/DeepPrint/fvc_2004_db2_b.csv", "deep_print_fvc_2004_db2_b"),
    ("../scores/DeepPrint/fvc_2004_db3_a.csv", "deep_print_fvc_2004_db3_a"),
    ("../scores/DeepPrint/fvc_2004_db3_b.csv", "deep_print_fvc_2004_db3_b"),
    ("../scores/DeepPrint/fvc_2004_db4_a.csv", "deep_print_fvc_2004_db4_a"),
    ("../scores/DeepPrint/fvc_2004_db4_b.csv", "deep_print_fvc_2004_db4_b"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db1_a.csv", "deep_print_priorenh_fvc_2000_db1_a"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db1_b.csv", "deep_print_priorenh_fvc_2000_db1_b"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db2_a.csv", "deep_print_priorenh_fvc_2000_db2_a"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db2_b.csv", "deep_print_priorenh_fvc_2000_db2_b"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db3_a.csv", "deep_print_priorenh_fvc_2000_db3_a"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db3_b.csv", "deep_print_priorenh_fvc_2000_db3_b"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db4_a.csv", "deep_print_priorenh_fvc_2000_db4_a"),
    ("../scores/DeepPrint_priorenh/fvc_2000_db4_b.csv", "deep_print_priorenh_fvc_2000_db4_b"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db1_a.csv", "deep_print_priorenh_fvc_2002_db1_a"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db1_b.csv", "deep_print_priorenh_fvc_2002_db1_b"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db2_a.csv", "deep_print_priorenh_fvc_2002_db2_a"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db2_b.csv", "deep_print_priorenh_fvc_2002_db2_b"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db3_a.csv", "deep_print_priorenh_fvc_2002_db3_a"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db3_b.csv", "deep_print_priorenh_fvc_2002_db3_b"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db4_a.csv", "deep_print_priorenh_fvc_2002_db4_a"),
    ("../scores/DeepPrint_priorenh/fvc_2002_db4_b.csv", "deep_print_priorenh_fvc_2002_db4_b"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db1_a.csv", "deep_print_priorenh_fvc_2004_db1_a"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db1_b.csv", "deep_print_priorenh_fvc_2004_db1_b"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db2_a.csv", "deep_print_priorenh_fvc_2004_db2_a"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db2_b.csv", "deep_print_priorenh_fvc_2004_db2_b"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db3_a.csv", "deep_print_priorenh_fvc_2004_db3_a"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db3_b.csv", "deep_print_priorenh_fvc_2004_db3_b"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db4_a.csv", "deep_print_priorenh_fvc_2004_db4_a"),
    ("../scores/DeepPrint_priorenh/fvc_2004_db4_b.csv", "deep_print_priorenh_fvc_2004_db4_b"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db1_a.csv", "deep_print_unetenh_fvc_2000_db1_a"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db1_b.csv", "deep_print_unetenh_fvc_2000_db1_b"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db2_a.csv", "deep_print_unetenh_fvc_2000_db2_a"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db2_b.csv", "deep_print_unetenh_fvc_2000_db2_b"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db3_a.csv", "deep_print_unetenh_fvc_2000_db3_a"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db3_b.csv", "deep_print_unetenh_fvc_2000_db3_b"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db4_a.csv", "deep_print_unetenh_fvc_2000_db4_a"),
    ("../scores/DeepPrint_unetenh/fvc_2000_db4_b.csv", "deep_print_unetenh_fvc_2000_db4_b"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db1_a.csv", "deep_print_unetenh_fvc_2002_db1_a"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db1_b.csv", "deep_print_unetenh_fvc_2002_db1_b"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db2_a.csv", "deep_print_unetenh_fvc_2002_db2_a"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db2_b.csv", "deep_print_unetenh_fvc_2002_db2_b"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db3_a.csv", "deep_print_unetenh_fvc_2002_db3_a"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db3_b.csv", "deep_print_unetenh_fvc_2002_db3_b"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db4_a.csv", "deep_print_unetenh_fvc_2002_db4_a"),
    ("../scores/DeepPrint_unetenh/fvc_2002_db4_b.csv", "deep_print_unetenh_fvc_2002_db4_b"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db1_a.csv", "deep_print_unetenh_fvc_2004_db1_a"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db1_b.csv", "deep_print_unetenh_fvc_2004_db1_b"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db2_a.csv", "deep_print_unetenh_fvc_2004_db2_a"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db2_b.csv", "deep_print_unetenh_fvc_2004_db2_b"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db3_a.csv", "deep_print_unetenh_fvc_2004_db3_a"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db3_b.csv", "deep_print_unetenh_fvc_2004_db3_b"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db4_a.csv", "deep_print_unetenh_fvc_2004_db4_a"),
    ("../scores/DeepPrint_unetenh/fvc_2004_db4_b.csv", "deep_print_unetenh_fvc_2004_db4_b"),
    ("../scores/FLARE/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db1_a"),
    ("../scores/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db1_a"),
    ("../scores/FLARE/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db1_b"),
    ("../scores/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db1_b"),
    ("../scores/FLARE/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db2_a"),
    ("../scores/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db2_a"),
    ("../scores/FLARE/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db2_b"),
    ("../scores/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db2_b"),
    ("../scores/FLARE/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db3_a"),
    ("../scores/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db3_a"),
    ("../scores/FLARE/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db3_b"),
    ("../scores/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db3_b"),
    ("../scores/FLARE/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db4_a"),
    ("../scores/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db4_a"),
    ("../scores/FLARE/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2000_db4_b"),
    ("../scores/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2000_db4_b"),
    ("../scores/FLARE/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db1_a"),
    ("../scores/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db1_a"),
    ("../scores/FLARE/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db1_b"),
    ("../scores/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db1_b"),
    ("../scores/FLARE/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db2_a"),
    ("../scores/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db2_a"),
    ("../scores/FLARE/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db2_b"),
    ("../scores/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db2_b"),
    ("../scores/FLARE/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db3_a"),
    ("../scores/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db3_a"),
    ("../scores/FLARE/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db3_b"),
    ("../scores/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db3_b"),
    ("../scores/FLARE/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db4_a"),
    ("../scores/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db4_a"),
    ("../scores/FLARE/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2002_db4_b"),
    ("../scores/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2002_db4_b"),
    ("../scores/FLARE/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db1_a"),
    ("../scores/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db1_a"),
    ("../scores/FLARE/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db1_b"),
    ("../scores/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db1_b"),
    ("../scores/FLARE/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db2_a"),
    ("../scores/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db2_a"),
    ("../scores/FLARE/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db2_b"),
    ("../scores/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db2_b"),
    ("../scores/FLARE/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db3_a"),
    ("../scores/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db3_a"),
    ("../scores/FLARE/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db3_b"),
    ("../scores/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db3_b"),
    ("../scores/FLARE/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db4_a"),
    ("../scores/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db4_a"),
    ("../scores/FLARE/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_regression_pose_fvc_2004_db4_b"),
    ("../scores/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_voting_pose_fvc_2004_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2000_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2000_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2002_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2002_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db1_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db1_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db2_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db2_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db3_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db3_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db4_a"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_priorenh_regression_pose_fvc_2004_db4_b"),
    ("../scores/FLARE_PRIORENH/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_priorenh_voting_pose_fvc_2004_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2000_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2000_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2002_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2002_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB1_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db1_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB1_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db1_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB2_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db2_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB2_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db2_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB3_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db3_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB3_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db3_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB4_A/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db4_a"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB4_B/FDD_feat_RegressionPose/score_FDD.csv", "flare_unetenh_regression_pose_fvc_2004_db4_b"),
    ("../scores/FLARE_UNETENH/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "flare_unetenh_voting_pose_fvc_2004_db4_b"),
)

DEFAULT_FILE_NAME_MAP: dict[str, str] = {
    file_path: file_name for file_path, file_name in DEFAULT_FILES
}

RESOLVED_DEFAULT_FILE_NAME_MAP: dict[Path, str] = {
    (SCRIPT_DIR / file_path).resolve(): file_name
    for file_path, file_name in DEFAULT_FILES
}


def normalize_color(
    color: tuple[float, float, float] | list[float] | str,
) -> tuple[float, float, float] | str:
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        if any(channel > 1.0 for channel in color[:3]):
            return (
                float(color[0]) / 255.0,
                float(color[1]) / 255.0,
                float(color[2]) / 255.0,
            )
        return (float(color[0]), float(color[1]), float(color[2]))
    return color


def get_file_identifier(input_file: str | Path, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name

    file_string = str(input_file)
    if file_string in DEFAULT_FILE_NAME_MAP:
        return DEFAULT_FILE_NAME_MAP[file_string]

    resolved_path = resolve_existing_file(input_file)
    if resolved_path in RESOLVED_DEFAULT_FILE_NAME_MAP:
        return RESOLVED_DEFAULT_FILE_NAME_MAP[resolved_path]

    stem = resolved_path.stem
    if stem.endswith("_metrics"):
        return stem[:-8]
    return stem


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{suffix}'. Expected .parquet or .csv")


def save_dataframe(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    float_format: str | None = f"%.{DECIMAL_PLACES}f",
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataframe.to_parquet(path, index=False)
    elif suffix == ".csv":
        dataframe.to_csv(path, index=False, float_format=float_format)
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


def derive_metrics_path(
    input_path: str | Path,
    output_name: str | None = None,
    output_format: str | None = None,
    output_directory: Path = METRICS_DIR,
) -> Path:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if output_format is not None:
        clean_format = output_format.lstrip(".").lower()
        target_suffix = f".{clean_format}"
    elif suffix in (".parquet", ".csv"):
        target_suffix = suffix
    else:
        target_suffix = ".csv"

    identifier = get_file_identifier(input_path, output_name)
    return output_directory / f"{identifier}{target_suffix}"


def derive_histogram_path(
    input_path: str | Path,
    output_name: str | None = None,
    output_directory: Path = HISTOGRAMS_DIR,
    extension: str = ".png",
) -> Path:
    identifier = get_file_identifier(input_path, output_name)
    target_extension = extension if extension.startswith(".") else f".{extension}"
    return output_directory / f"{identifier}{target_extension}"


derive_metrics_csv_path = derive_metrics_path


def extract_user_scores(
    comparisons: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    users = comparisons["user_1"].unique()
    user_groups = comparisons.groupby("user_1")
    user_scores: list[tuple[np.ndarray, np.ndarray]] = []

    for user in users:
        group = user_groups.get_group(user)
        genuine = group.loc[group["user_2"] == user, "score"].to_numpy()
        impostor = group.loc[group["user_2"] != user, "score"].to_numpy()
        user_scores.append((np.sort(genuine), np.sort(impostor)))

    return user_scores


def evaluate_threshold_group(
    threshold_group: np.ndarray,
    user_scores: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float]]:
    chunk_size = len(threshold_group)
    if not user_scores:
        return [
            {"frr": 0.0, "far": 0.0, "threshold": float(threshold_group[index])}
            for index in range(chunk_size)
        ]

    user_frr: list[np.ndarray] = []
    user_far: list[np.ndarray] = []

    for genuine_scores, impostor_scores in user_scores:
        total_genuine = len(genuine_scores)
        total_impostor = len(impostor_scores)

        false_rejections = np.searchsorted(genuine_scores, threshold_group, side="left")
        false_acceptances = total_impostor - np.searchsorted(impostor_scores, threshold_group, side="left")

        user_frr.append(
            false_rejections / total_genuine
            if total_genuine > 0
            else np.zeros(chunk_size, dtype=float)
        )
        user_far.append(
            false_acceptances / total_impostor
            if total_impostor > 0
            else np.zeros(chunk_size, dtype=float)
        )

    mean_frr = np.mean(user_frr, axis=0)
    mean_far = np.mean(user_far, axis=0)

    return [
        {
            "frr": float(mean_frr[index]),
            "far": float(mean_far[index]),
            "threshold": float(threshold_group[index]),
        }
        for index in range(chunk_size)
    ]


def evaluate_thresholds_in_parallel(
    sample_thresholds: np.ndarray,
    user_scores: list[tuple[np.ndarray, np.ndarray]],
    group_size: int = THRESHOLD_GROUP_SIZE,
    max_workers: int | None = None,
    executor_type: str = "process",
) -> list[dict[str, float]]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    threshold_groups = [
        sample_thresholds[index : index + group_size]
        for index in range(0, len(sample_thresholds), group_size)
    ]

    if not threshold_groups:
        return []

    if executor_type == "thread":
        executor_class = ThreadPoolExecutor
    elif executor_type == "process":
        executor_class = ProcessPoolExecutor
    else:
        raise ValueError(
            f"Unsupported executor_type '{executor_type}'. Expected 'process' or 'thread'."
        )

    results: list[dict[str, float]] = []
    with executor_class(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_threshold_group, group, user_scores)
            for group in threshold_groups
        ]
        for future in futures:
            results.extend(future.result())

    return results


def calculate_metrics(
    input_file: str | Path,
    output_file: str | Path | None = None,
    output_name: str | None = None,
    output_format: str | None = None,
    group_size: int = THRESHOLD_GROUP_SIZE,
    max_workers: int | None = None,
    executor_type: str = "process",
) -> None:
    resolved_path = resolve_existing_file(input_file)
    if not resolved_path.is_file() or resolved_path.suffix.lower() not in (".csv", ".parquet"):
        print(f"Skipping invalid input file: {input_file}")
        return

    if output_file is None:
        out_path = derive_metrics_path(
            resolved_path,
            output_name=output_name,
            output_format=output_format,
        )
    else:
        out_path = (SCRIPT_DIR / output_file).resolve() if not Path(output_file).is_absolute() else Path(output_file)

    comparisons = load_dataframe(resolved_path)
    if comparisons.empty or "score" not in comparisons or "user_1" not in comparisons or "user_2" not in comparisons:
        print(f"Skipping empty or invalid dataframe: {resolved_path}")
        return

    thresholds = comparisons["score"].unique()
    start_threshold = thresholds.max()
    end_threshold = thresholds.min()
    sample_thresholds = np.linspace(start_threshold, end_threshold, 1000)

    user_scores = extract_user_scores(comparisons)
    results = evaluate_thresholds_in_parallel(
        sample_thresholds,
        user_scores,
        group_size=group_size,
        max_workers=max_workers,
        executor_type=executor_type,
    )

    result = pd.DataFrame(results).round(DECIMAL_PLACES)
    save_dataframe(result, out_path)
    print(f"Metrics saved to: {out_path}")


def plot_histogram(
    input_file: str | Path,
    output_file: str | Path | None = None,
    output_name: str | None = None,
    bin_width: float = BIN_WIDTH,
    genuine_color: tuple[float, float, float] | list[float] | str = HISTOGRAM_GENUINE_COLOR,
    impostor_color: tuple[float, float, float] | list[float] | str = HISTOGRAM_IMPOSTOR_COLOR,
    alpha: float = HISTOGRAM_ALPHA,
) -> None:
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

    if output_file is None:
        out_png = derive_histogram_path(resolved_path, output_name=output_name)
    else:
        out_png = (SCRIPT_DIR / output_file).resolve() if not Path(output_file).is_absolute() else Path(output_file)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    normalized_genuine_color = normalize_color(genuine_color)
    normalized_impostor_color = normalize_color(impostor_color)
    title_text = output_name or get_file_identifier(resolved_path)

    plt.figure(figsize=(10, 6))
    plt.hist(
        genuine,
        bins=bin_edges,
        alpha=alpha,
        label="Mesmo usuário",
        color=normalized_genuine_color,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.hist(
        impostor,
        bins=bin_edges,
        alpha=alpha,
        label="Usuários diferentes",
        color=normalized_impostor_color,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Score")
    plt.ylabel("Quantidade")
    plt.title(title_text)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Histogram saved to: {out_png}")


def find_min_diff(metrics_file: str | Path, result_name: str | None = None) -> dict | None:
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

    frr = round(float(best_row["frr"]), DECIMAL_PLACES)
    far = round(float(best_row["far"]), DECIMAL_PLACES)
    err = round((frr + far) / 2.0, DECIMAL_PLACES)
    threshold = round(float(best_row["threshold"]), DECIMAL_PLACES)

    filename = result_name if result_name is not None else get_file_identifier(resolved_path)

    print(f"File: {resolved_path.name}")
    print(f"Row with minimal |frr - far| (diff = {best_diff:.10f}):")
    print(f"  far = {far:.{DECIMAL_PLACES}f}")
    print(f"  frr = {frr:.{DECIMAL_PLACES}f}")
    print(f"  err = {err:.{DECIMAL_PLACES}f}")
    print(f"  threshold = {threshold:.{DECIMAL_PLACES}f}")
    return {
        "file": filename,
        "far": far,
        "frr": frr,
        "err": err,
        "threshold": threshold,
    }


def run_metrics_command(
    files: tuple[tuple[str, str], ...] | list,
    output_format: str | None = None,
    group_size: int = THRESHOLD_GROUP_SIZE,
    max_workers: int | None = None,
    executor_type: str = "process",
) -> None:
    for item in files:
        if isinstance(item, (tuple, list)):
            input_file = item[0]
            output_name = item[1] if len(item) > 1 and item[1] else None
        else:
            input_file, output_name = item, None

        print(f"Calculating metrics for: {input_file}")
        calculate_metrics(
            input_file,
            output_name=output_name,
            output_format=output_format,
            group_size=group_size,
            max_workers=max_workers,
            executor_type=executor_type,
        )


def run_histogram_command(
    files: tuple[tuple[str, str], ...] | list,
    bin_width: float = BIN_WIDTH,
    genuine_color: tuple[float, float, float] | list[float] | str = HISTOGRAM_GENUINE_COLOR,
    impostor_color: tuple[float, float, float] | list[float] | str = HISTOGRAM_IMPOSTOR_COLOR,
    alpha: float = HISTOGRAM_ALPHA,
) -> None:
    for item in files:
        if isinstance(item, (tuple, list)):
            input_file = item[0]
            output_name = item[1] if len(item) > 1 and item[1] else None
        else:
            input_file, output_name = item, None

        print(f"Plotting histogram for: {input_file}")
        plot_histogram(
            input_file,
            output_name=output_name,
            bin_width=bin_width,
            genuine_color=genuine_color,
            impostor_color=impostor_color,
            alpha=alpha,
        )


def run_min_diff_command(
    files: tuple[tuple[str, str], ...] | list,
    output_file: str | Path = RESULTS_DIR / "min_diff.csv",
) -> None:
    results = []

    for item in files:
        if isinstance(item, (tuple, list)):
            input_file = item[0]
            result_name = item[1] if len(item) > 1 and item[1] else None
        else:
            input_file, result_name = item, None

        identifier = get_file_identifier(input_file, result_name)

        metrics_path = METRICS_DIR / f"{identifier}.csv"
        resolved_metrics_path = resolve_existing_file(metrics_path)

        if not resolved_metrics_path.is_file():
            parquet_metrics_path = METRICS_DIR / f"{identifier}.parquet"
            resolved_metrics_path = resolve_existing_file(parquet_metrics_path)

        if not resolved_metrics_path.is_file():
            direct_resolved = resolve_existing_file(input_file)
            if direct_resolved.is_file() and ("_metrics" in direct_resolved.stem or direct_resolved.parent == METRICS_DIR):
                resolved_metrics_path = direct_resolved
            else:
                old_metrics_path = direct_resolved.with_name(f"{direct_resolved.stem}_metrics.csv")
                resolved_metrics_path = resolve_existing_file(old_metrics_path)

        print(f"Finding min diff for: {resolved_metrics_path.name}")
        row_dict = find_min_diff(resolved_metrics_path, result_name=identifier)
        if row_dict is not None:
            results.append(row_dict)

    if results:
        out_path = (SCRIPT_DIR / output_file).resolve() if not Path(output_file).is_absolute() else Path(output_file)
        results_dataframe = pd.DataFrame(
            results, columns=["file", "far", "frr", "err", "threshold"]
        ).round(DECIMAL_PLACES)
        save_dataframe(results_dataframe, out_path)
        print(f"Min diff results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biometric score analysis tool combining metrics calculation, histogram plotting, and minimum difference analysis.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    metrics_parser = subparsers.add_parser("metrics", help="Calculate FRR and FAR metrics across thresholds.")
    metrics_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    metrics_parser.add_argument("--format", choices=["csv", "parquet"], default=None, help="Output format for metrics files (defaults to input file format).")
    metrics_parser.add_argument("--group-size", type=int, default=THRESHOLD_GROUP_SIZE, help="Threshold chunk size for parallel processing (default: 200).")
    metrics_parser.add_argument("--workers", type=int, default=None, help="Number of worker threads or processes.")
    metrics_parser.add_argument("--executor", choices=["process", "thread"], default="process", help="Parallel execution backend ('process' or 'thread').")

    histogram_parser = subparsers.add_parser("histogram", help="Plot score distribution histograms.")
    histogram_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    histogram_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")
    histogram_parser.add_argument("--alpha", type=float, default=HISTOGRAM_ALPHA, help=f"Transparency (alpha) for histogram bars (default: {HISTOGRAM_ALPHA}).")
    histogram_parser.add_argument("--genuine-color", type=float, nargs=3, default=list(HISTOGRAM_GENUINE_COLOR), metavar=("R", "G", "B"), help=f"RGB values for genuine scores (default: {HISTOGRAM_GENUINE_COLOR}).")
    histogram_parser.add_argument("--impostor-color", type=float, nargs=3, default=list(HISTOGRAM_IMPOSTOR_COLOR), metavar=("R", "G", "B"), help=f"RGB values for impostor scores (default: {HISTOGRAM_IMPOSTOR_COLOR}).")

    min_diff_parser = subparsers.add_parser("min-diff", help="Find row with minimal |frr - far| from metrics files.")
    min_diff_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files or metrics files (.csv or .parquet) to process.")
    min_diff_parser.add_argument("--output", default=str(RESULTS_DIR / "min_diff.csv"), help="Output file path for min diff results (.csv or .parquet).")

    all_parser = subparsers.add_parser("all", help="Run metrics, histogram, and min-diff analysis on target files.")
    all_parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Score files (.csv or .parquet) to process.")
    all_parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help="Bin width for histogram.")
    all_parser.add_argument("--alpha", type=float, default=HISTOGRAM_ALPHA, help=f"Transparency (alpha) for histogram bars (default: {HISTOGRAM_ALPHA}).")
    all_parser.add_argument("--genuine-color", type=float, nargs=3, default=list(HISTOGRAM_GENUINE_COLOR), metavar=("R", "G", "B"), help=f"RGB values for genuine scores (default: {HISTOGRAM_GENUINE_COLOR}).")
    all_parser.add_argument("--impostor-color", type=float, nargs=3, default=list(HISTOGRAM_IMPOSTOR_COLOR), metavar=("R", "G", "B"), help=f"RGB values for impostor scores (default: {HISTOGRAM_IMPOSTOR_COLOR}).")
    all_parser.add_argument("--format", choices=["csv", "parquet"], default=None, help="Output format for metrics files (defaults to input file format).")
    all_parser.add_argument("--output", default=str(RESULTS_DIR / "min_diff.csv"), help="Output file path for min diff results (.csv or .parquet).")
    all_parser.add_argument("--group-size", type=int, default=THRESHOLD_GROUP_SIZE, help="Threshold chunk size for parallel processing (default: 200).")
    all_parser.add_argument("--workers", type=int, default=None, help="Number of worker threads or processes.")
    all_parser.add_argument("--executor", choices=["process", "thread"], default="process", help="Parallel execution backend ('process' or 'thread').")

    args = parser.parse_args()

    if args.command in ("metrics", "histogram", "min-diff", "all"):
        files: tuple[tuple[str, str], ...] = tuple(
            (item, get_file_identifier(item)) if isinstance(item, str) else tuple(item)
            for item in args.files
        )

        if args.command == "metrics":
            run_metrics_command(
                files,
                output_format=args.format,
                group_size=args.group_size,
                max_workers=args.workers,
                executor_type=args.executor,
            )
        elif args.command == "histogram":
            run_histogram_command(
                files,
                bin_width=args.bin_width,
                genuine_color=tuple(args.genuine_color),
                impostor_color=tuple(args.impostor_color),
                alpha=args.alpha,
            )
        elif args.command == "min-diff":
            run_min_diff_command(files, args.output)
        elif args.command == "all":
            run_metrics_command(
                files,
                output_format=args.format,
                group_size=args.group_size,
                max_workers=args.workers,
                executor_type=args.executor,
            )
            run_histogram_command(
                files,
                bin_width=args.bin_width,
                genuine_color=tuple(args.genuine_color),
                impostor_color=tuple(args.impostor_color),
                alpha=args.alpha,
            )
            run_min_diff_command(files, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

