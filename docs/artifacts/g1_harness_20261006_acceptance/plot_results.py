"""Create standalone figures from archived measurements; no network or actuation."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def degrees(value):
    return math.degrees(math.atan2(math.sin(value), math.cos(value)))


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.root, args.root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    reach = rows(root / "bottle-reach-final/samples.jsonl")
    ax = axes[0, 0]
    for side, color in [("left", "#4169a0"), ("right", "#d58421")]:
        t = [r["elapsed_s"] for r in reach]
        gaps = [100 * r["hand_bottle_collision_distance_m"][side] for r in reach]
        ax.plot(t, gaps, label=f"{side.capitalize()} hand", color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Time since manipulation start (s)", ylabel="Sampled geometry gap (cm)",
           title="The simulated hand stays short of the bottle")
    ax.text(.04, .07, "Closest sampled right-hand gap: 19.84 cm\n10 Hz reconstruction; bottle orientation assumed",
            transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)

    audit = read(root / "control-mode-audit/audit.json")
    ax = axes[0, 1]
    names = ["OFF", "POSE", "PLANNER", "PLANNER_FROZEN_UPPER_BODY"]
    labels = ["Off", "Pose", "Planner", "Planner, upper body frozen"]
    counts = [audit["mode_counts"][n] for n in names]
    ax.barh(labels, counts, color=["#999999", "#4169a0", "#68a27b", "#d58421"])
    for i, count in enumerate(counts):
        ax.text(count + 2000, i, f"{count:,} ({100 * count / audit['frames']:.1f}%)", va="center", fontsize=9)
    ax.set(xlim=(0, 228000), xlabel="Recorded source frames", title="Control mode does not identify the task")
    ax.text(.02, -.23, "174 retained episodes; future plan has 0 accepted training segments.\nCurrent enum used; historical enum revision is unverified.",
            transform=ax.transAxes, fontsize=9)

    for ax, name, title in [(axes[1, 0], "native-repeat-baseline", "Turn immediately after walking"),
                            (axes[1, 1], "native-repeat-pause", "Turn after a one-second stationary request")]:
        turns = [r for r in rows(root / name / "native-trace.jsonl") if r["event"] == "turn_sample"]
        active = [r for r in turns if r["phase"] == "TURNING"]
        result = read(root / name / "result.json")
        origin = turns[0]["goal_yaw_rad"] - math.radians(result["turn_angle_deg"])
        t0 = turns[0]["at"]
        for key, label, color in [("reference_yaw_rad", "Facing reference", "#4169a0"),
                                   ("measured_yaw_rad", "Measured heading", "#d58421")]:
            # Terminal interrupted reference is retained internal state, not a wire command.
            samples = active if key == "reference_yaw_rad" else turns
            ax.plot([r["at"] - t0 for r in samples],
                    [degrees(r[key] - origin) for r in samples], label=label, color=color)
        ax.axhline(-15, linestyle="--", color="black", label="Requested turn")
        ax.axhspan(-18, -12, color="#68a27b", alpha=.16, label="3-degree tolerance")
        ax.axvline(turns[-1]["at"] - t0, color="#b44b4b", linestyle=":")
        ax.set(xlabel="Time since first logged turn sample (s)", ylabel="Heading change (degrees)",
               title=title, ylim=(-20, 8))
        reason = "Lead/rate guard" if name.endswith("baseline") else "10-second deadline"
        ax.text(.02, .05, f"{reason}: turn failed\nFresh hold acknowledged in {result['hold_ack_elapsed_s']:.3f} s",
                transform=ax.transAxes, fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("G1 harness: measured gaps remain — simulation evidence only", fontsize=16)
    save(fig, output, "acceptance-overview")

    camera = read(root / "camera-proxy/result.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, name, title in [(axes[0], "existing", "Existing simulator camera"),
                            (axes[1], "urdf_mount_proxy", "D435 URDF mount proxy")]:
        ax.imshow(plt.imread(root / "camera-proxy" / name / "ego_view.png"))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Same robot pose and physics; camera changes by {100 * camera['mount_translation_difference_m']:.2f} cm / "
                 f"{camera['mount_rotation_difference_deg']:.2f} degrees\n"
                 "Factory RGB intrinsics and optical-to-robot transform still unavailable", fontsize=13)
    save(fig, output, "camera-mount-comparison")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    examples = [("episode-78-0.jpg", "Bottle lying on the destination stool · source 78, 39 s"),
                ("episode-52-1.jpg", "Bottle still held · source 52, 54.6 s"),
                ("episode-155-0.jpg", "Cup action under bottle instruction · source 155, 6 s"),
                ("episode-159-1.jpg", "Human setup · source 159, 7 s")]
    for ax, (name, title) in zip(axes.flat, examples):
        ax.imshow(plt.imread(root / "vision-recorded/frames" / name))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Recorded-frame examples: sampled review, no new accepted episode labels", fontsize=14)
    save(fig, output, "recorded-review-examples")
    assert all(np.isfinite(r["hand_bottle_collision_distance_m"]["right"]) for r in reach)
    print(f"Saved three PNG/PDF figure pairs under {output}")


if __name__ == "__main__":
    main()
