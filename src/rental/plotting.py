"""Helpers for plotting Persian text with matplotlib."""
import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from bidi.algorithm import get_display


def fa(text) -> str:
    """Reshape + reorder Persian text so it renders correctly in matplotlib."""
    return get_display(arabic_reshaper.reshape(str(text)))


def setup():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
