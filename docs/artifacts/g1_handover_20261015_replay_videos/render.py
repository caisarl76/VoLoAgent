"""Render saved ground-truth physics (20 Hz qpos) of a trial to MP4 with a caption."""
import gzip, json, subprocess, sys
from pathlib import Path
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 960, 540, 20
VIEWS = {  # lookat, distance, azimuth, elevation
    "side": ((0.20, -0.15, 0.82), 1.35, 90.0, -8.0),
    "wide": ((0.15, -0.1, 0.65), 2.2, 75.0, -10.0),
}


def frames(run):
    gt = [json.loads(s) for s in gzip.open(run / "ground-truth.jsonl.gz", "rt")]
    sim = np.array([g["sim_time"] for g in gt])
    start = int(np.where(np.diff(sim) < 0)[0][-1] + 1) if np.any(np.diff(sim) < 0) else 0
    return gt[start:]


def render(run, out, caption, view="side", t0=None, t1=None, still=None):
    run = Path(run)
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = VIEWS[view]
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    font = ImageFont.load_default(size=18)
    rows = [g for g in frames(run) if (t0 is None or g["sim_time"] >= t0) and (t1 is None or g["sim_time"] <= t1)]
    if still is not None:
        rows = [min(rows, key=lambda g: abs(g["sim_time"] - still))]
    proc = None if still is not None else subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    base = rows[0]["sim_time"]
    for g in rows:
        data.qpos[:] = g["qpos"]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam, opt)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, W, 36), fill=(0, 0, 0))
        draw.text((10, 6), f"{caption}   t={g['sim_time'] - base:5.2f} s", fill=(255, 255, 255), font=font)
        if still is not None:
            img.save(out)
            return
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    run, out, caption = sys.argv[1:4]
    kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    render(run, out, caption, **kw)
