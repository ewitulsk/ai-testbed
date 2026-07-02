"""Shared IO + plotting style for consensus-seismology analysis scripts."""
import os, sys, json, gzip
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)
os.makedirs(RESULTS, exist_ok=True)

# ---- palette (colour-blind-safe, works light/dark) ----
INK = "#1a1a2e"
GRID = "#d9dce3"
C_DATA = "#2a6f97"     # empirical / data
C_MODEL = "#c1121f"    # power-law / theory
C_ALT = "#8d99ae"      # alternative model (exponential)
C_ACCENT = "#e09f3e"   # highlight
C_STRAIN = "#5a189a"   # strain index

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "font.size": 10.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def dump_json(obj, name):
    path = os.path.join(RESULTS, name)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    print(f"  wrote {path}")
    return path


def load_eth_blocks():
    import pandas as pd
    path = os.path.join(DATA, "eth_blocks.csv.gz")
    return pd.read_csv(path)


def load_eth_uncles():
    import pandas as pd
    path = os.path.join(DATA, "eth_uncles.csv.gz")
    return pd.read_csv(path)


def ccdf(x):
    """Empirical complementary CDF points (x sorted asc, P(X>=x))."""
    x = np.sort(np.asarray(x))
    n = len(x)
    vals, idx = np.unique(x, return_index=True)
    surv = 1.0 - idx / n
    return vals, surv
