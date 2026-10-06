"""Show the frozen appearance control beside a recorded training view."""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    args = parser.parse_args()
    camera = cv2.VideoCapture(str(args.video))
    camera.set(cv2.CAP_PROP_POS_FRAMES, 368)
    ok, recorded = camera.read()
    camera.release()
    if not ok:
        raise ValueError('Cannot decode recorded frame 368')
    cv2.imwrite(str(args.root/'appearance-control/recorded-18-frame-368.jpg'), recorded)
    cyan = cv2.imread(str(args.root/'appearance-control/cyan/ego_view.png'))
    clear = cv2.imread(str(args.root/'appearance-control/clear_green_cap/ego_view.png'))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, frame, title in zip(axes, [recorded, cyan, clear], [
        'Training episode 18: 7.36 s', 'Measured bottle: cyan cylinder',
        'Same physics: clear cylinder + visual cap',
    ]):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.suptitle('Appearance control: bottle size/mass/collisions held fixed; full scene still approximate')
    for suffix in ['png', 'pdf']:
        fig.savefig(args.root/f'appearance-comparison.{suffix}', dpi=160)


if __name__ == '__main__':
    main()
