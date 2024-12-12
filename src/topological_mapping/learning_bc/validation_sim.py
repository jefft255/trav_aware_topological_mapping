import subprocess
import time
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def run_simulator_once(output_dir, run_id, timeout, only_front):
    script_path = (
        Path(__file__).parent.parent.parent.parent / "scripts" / "test_bc_narval.sh"
    )
    args_list = ["bash", str(script_path), str(output_dir), str(run_id)]
    if only_front:
        args_list.append("only_front")
    # Timeout for initial process launch
    subprocess.run(
        args_list,
        check=False,
        timeout=timeout,
    )
    file_found = False
    time_begin = time.time()
    # Then timeout for finding the actual result file; the process aboves terminates before everything is done
    while not file_found and time.time() - time_begin < timeout:
        time.sleep(5)
        if (output_dir / f"test_run_{run_id}" / "test_bc_results.csv").exists():
            file_found = True
    if not file_found:
        raise TimeoutError("Timeout while waiting for simulator to finish")


def validation_simulator(output_dir, cfg, logger, epoch):
    for i in range(cfg.n_simulations):
        run_simulator_once(output_dir, i, cfg.timeout_simulator, cfg.only_front)

    success_rates = []

    for i in range(cfg.n_simulations):
        with open(output_dir / f"test_run_{i}" / "test_bc_results.csv", "r") as f:
            first_line = f.readline().strip("\n")
            success_rate = float(first_line.split(",")[1])
            success_rates.append(success_rate)
    logger.add_scalar("successrate_mean", np.mean(np.array(success_rates)), epoch)
    logger.add_scalar("successrate_std", np.std(np.array(success_rates)), epoch)

    fig = plt.figure(figsize=(3.5, 3.5), dpi=300)
    ax = fig.gca()
    sns.violinplot(
        y=success_rates, x=[0] * len(success_rates), ax=ax, orient="v", inner="box"
    )
    ax.set_ylim(0, 1)
    logger.add_figure("successrate_violin", fig, epoch)
    return np.mean(np.array(success_rates))
