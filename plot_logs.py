import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_all_logs(log_dir="logs"):
    pattern = os.path.join(log_dir, "logs_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No log files found matching {pattern}")
    dfs = []
    for fp in files:
        dfs.append(pd.read_csv(fp))
    return pd.concat(dfs, ignore_index=True)

def coerce_numeric(df):
    for col in ["rand","rand_s","ent","ent_s","pos","pos_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_stats(df):
    stats = {}
    for key, gcol, scol in [
        ("random","rand","rand_s"),
        ("entropy","ent","ent_s"),
        ("positional","pos","pos_s")
    ]:
        g = df[gcol]
        s = df[scol]
        g_all = g[g.notna()].values

        g_success = g[(g.notna()) & (s == 1)].values

        stats[key] = {
            "guesses_all": g_all,
            "guesses_success": g_success,
            "mean_all": np.mean(g_all) if len(g_all) else np.nan,
            "mean_success": np.mean(g_success) if len(g_success) else np.nan,
        }
    return stats

def plot_hist(stats, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    strategies = [
        ("random","Random"),
        ("entropy","Entropy"),
        ("positional","Positional")
    ]

    # Determine max guess count for binning
    max_guess = 0
    for k,_ in strategies:
        if len(stats[k]["guesses_all"]) > 0:
            max_guess = max(max_guess, int(np.nanmax(stats[k]["guesses_all"])))
    if max_guess < 1:
        max_guess = 10

    bins = np.arange(1, max_guess+2) - 0.5

    for ax, (k,label) in zip(axes, strategies):
        g_all = stats[k]["guesses_all"]

        if len(g_all) > 0:
            ax.hist(g_all, bins=bins, edgecolor="black")

        mean_all = stats[k]["mean_all"]
        mean_success = stats[k]["mean_success"]

        if not np.isnan(mean_all):
            ax.axvline(mean_all, color="red", linestyle="-", linewidth=2,
                       label=f"Mean (all): {mean_all:.2f}")

        if not np.isnan(mean_success):
            ax.axvline(mean_success, color="blue", linestyle="--", linewidth=2,
                       label=f"Mean (success): {mean_success:.2f}")

        ax.set_title(f"{label} guess distribution")
        ax.set_ylabel("Count")
        ax.legend()

    axes[-1].set_xticks(np.arange(1, max_guess+1))
    axes[-1].set_xlabel("Number of guesses")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    log_dir = "logs"
    df = load_all_logs(log_dir)
    df = coerce_numeric(df)
    stats = compute_stats(df)

    out_path = os.path.join(log_dir, "summary_hist.png")
    plot_hist(stats, out_path)

    print(f"Histogram summary saved: {out_path}")

if __name__ == "__main__":
    main()
