import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

N_SIMULATIONS = 20
N_EXP = 10


def validation_simulator(output_dir):
    success_rates = []
    fractions = []
    fractions_means = []
    means = []

    for exp in range(N_EXP):
        x = str(exp * 0.1 + 0.1)[:3]
        x_mean = exp * 1 + 10
        folder = output_dir / str(exp)
        for i in range(N_SIMULATIONS):
            try:
                with open(folder / f"test_run_{i}" / "test_bc_results.csv", "r") as f:
                    print(str(folder / f"test_run_{i}" / "test_bc_results.csv"))
                    for line in f.readlines():
                        first_line = line.strip("\n")
                        try:
                            success_rate = float(first_line.split(",")[1])
                            print(f"{exp=}, {i=}, {success_rate=}")
                            break
                        except:
                            pass
                        success_rate = 0
            except FileNotFoundError:
                success_rate = 0
            success_rates.append(success_rate)
            fractions.append(x)
        means.append(np.mean(success_rates[-N_SIMULATIONS:]))
        fractions_means.append(x_mean)
        print(success_rates)

    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig = plt.figure(figsize=(3.5, 2.5), dpi=300)
    ax = fig.gca()
    print(fractions)
    sns.violinplot(
        y=success_rates, x=fractions, ax=ax, orient="v", inner="box", linewidth=0.4
    )
    ax.set_xlabel("Fraction of dataset used")
    ax.set_ylabel("Success rate")
    # sns.lineplot(x=fractions_means, y=means, ax=ax)
    ax.set_ylim(0, 1)
    plt.tight_layout(pad=0.1)
    plt.savefig("perf.pdf")


if __name__ == "__main__":
    validation_simulator(Path(sys.argv[1]))
