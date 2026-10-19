"""Render saved corrected-run physics; no new simulation or control."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path[:0] = [str(ROOT.parent / "g1_handover_20261012_acceptance"),
               str(ROOT.parent / "g1_handover_20261013_lift")]
import render_ready
from geometry import closest_bounds_gap

render_ready.ROOT = ROOT
render_ready.RUN_NAME = "decoder-trace"
render_ready.MOTION_LABEL = "Corrected reference guard | SIMULATION"
render_ready.CLEARANCE_LABEL = "conservative desk gap"
render_ready.CLEARANCE_FUNCTION = lambda model, data: closest_bounds_gap(
    model, data, ["desk_top", *(f"desk_leg_{i}" for i in range(4))]
)["gap_m"]

if __name__ == "__main__":
    render_ready.main()
