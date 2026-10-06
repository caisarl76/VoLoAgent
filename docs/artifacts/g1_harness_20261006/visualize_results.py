"""Render saved measurements only; no policy, network or actuator calls."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).parent


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def save(fig, name):
    fig.savefig(ROOT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    analysis = read("policy-analysis/summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, path, title in [
        (axes[0, 0], "policy-replay/recorded-10s.jpg", "Recorded training episode 18 at 10 s"),
        (axes[0, 1], "bottle-measured-scene/ego_view.png", "Sample scene: 20 cm × 8 cm, 300 g bottle")]:
        axis.imshow(plt.imread(ROOT / path))
        axis.set_title(title)
        axis.axis("off")
    before, after = analysis["state_before"], analysis["state_after"]
    seconds = [row["seconds"] for row in before]
    axis = axes[1, 0]
    axis.plot(seconds, [row["max_absolute_state_delta"]["left_hand"] for row in before],
              "o-", color="#b84736", label="Previous preparation")
    axis.plot(seconds, [row["max_absolute_state_delta"]["left_hand"] for row in after],
              "o-", color="#208368", label="Measured fingers preserved")
    axis.set(xlabel="Time in recorded episode (s)", ylabel="Largest left-hand input error (rad)",
             title="Input mismatch fixed in all six sampled frames")
    axis.legend(frameon=False)
    truth = rows("bottle-physical-calibration-truth-control/bottle-ground-truth.jsonl")
    xyz = np.array([row["bottle_position"] for row in truth])
    axis = axes[1, 1]
    axis.add_patch(Rectangle((0.5-0.237, 0.2-0.3), 0.474, 0.6, color="#d8b582", alpha=0.5))
    axis.add_patch(Circle((0.35, -0.6), 0.25, color="#99bf71", alpha=0.6))
    axis.plot(xyz[:, 0], xyz[:, 1], color="#197c92", linewidth=2)
    axis.scatter(*xyz[0, :2], s=70, facecolors="none", edgecolors="#197c92", label="Bottle samples")
    axis.text(0.5, 0.4, "Source table", ha="center")
    axis.text(0.35, -0.6, "Intended\ngreen stool", ha="center", va="center")
    axis.set(xlim=(0, 0.85), ylim=(-0.95, 0.6), aspect="equal", xlabel="Sample-scene x (m)",
             ylabel="Sample-scene y (m)", title=f"{len(truth):,} samples: bottle stays on source")
    fig.suptitle("G1 diagnosis: input fixed; learned grasp still unvalidated", fontsize=16)
    fig.text(0.5, 0.015, "Offline replay is a training-fit check. Scene/camera remain provisional. Live monitor for this trial is scripted simulator truth.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "bottle-diagnosis")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    batch = read("turn-repeat/results.json")
    for axis, case in zip(axes.flat, batch["cases"]):
        result = case["result"]
        data = rows(f"turn-repeat/{case['case']}/turn-shadow.jsonl")
        at = np.array([row["at"] for row in data])
        initial = result["turn_initial_yaw_rad"]
        def delta(values):
            return np.degrees((np.array(values)-initial+math.pi) % (2*math.pi)-math.pi)
        goal = float(delta([result["turn_goal_yaw_rad"]])[0])
        axis.axhspan(goal-3, goal+3, color="#208368", alpha=0.15, label="±3° completion tolerance")
        axis.axhline(goal, color="#208368", linestyle=":", label="Physical heading goal")
        axis.plot(at-at[0], delta([row["measured_yaw_rad"] for row in data]), color="#176684", label="Measured heading")
        axis.plot(at-at[0], delta([row["facing_yaw_rad"] for row in data]), color="#b7752d", linestyle="--", label="Planner reference")
        error = math.degrees(result["turn_error_rad"])
        outcome = "Passed" if result["outcome"] == "passed" else "Interrupted: lead/rate conflict"
        axis.set_title(f"{case['case']} — {outcome}\nFinal error {error:.2f}°; {result['turn_elapsed_s']:.2f} s")
        axis.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Four fresh SONIC turn controls: 2 pass, 2 interrupt", fontsize=16)
    fig.supxlabel("Elapsed time from first turn sample (s)", y=0.005)
    fig.supylabel("Heading relative to turn entry (degrees)")
    fig.text(0.5, 0.045, "Real C++ SONIC + MuJoCo, adapter-driven. Limits unchanged: 10°/s, 5° lead, 3° for 0.5 s, 10 s deadline. DDS lo only.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.94))
    save(fig, "turn-repeat")


if __name__ == "__main__":
    main()
