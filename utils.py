# utils.py
import numpy as np
import pandas as pd
from math import degrees
import streamlit as st
import statsmodels.api as sm
from scipy.signal import savgol_filter
def humanize_label(label):
    """
    Converts a snake_case or camelCase string into Title Case.
    """
    if not label:
        return "N/A"
    return " ".join(word.capitalize() for word in label.replace('_', ' ').split())

def humanize_segment_label(segment):
    """
    Converts segment strings like 'segment_004' into 'Shot 4'.
    Removes any leading zeros.
    """
    if not segment:
        return "N/A"
    if segment.lower().startswith("segment_"):
        # Get the numeric part and remove leading zeros.
        num_part = segment.split("_", 1)[1].lstrip("0")
        if not num_part:
            num_part = "0"
        return "Shot " + num_part
    return segment


def add_time_column(df, fps=25):
    """
    Adds a 'time' column computed from the 'frame' column (assuming frame numbers are available).
    """
    if "frame" in df.columns:
        df["time"] = df["frame"] / fps
    else:
        df["time"] = None
    return df

