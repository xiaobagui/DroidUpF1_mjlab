"""Cut NPZ motion data by frame range.

Usage:
    # Cut frames 100 to 500
    python legged_lab/tools/cut_npz.py --npz input.npz --out output.npz --start 100 --end 500

    # Cut first 200 frames
    python legged_lab/tools/cut_npz.py --npz input.npz --out output.npz --end 200

    # Cut from frame 300 to end
    python legged_lab/tools/cut_npz.py --npz input.npz --out output.npz --start 300
"""

import argparse
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Cut NPZ motion data by frame range")
    parser.add_argument("--npz", type=str, required=True, help="Input NPZ file")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ file")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
    args = parser.parse_args()

    print(f"Loading: {args.npz}")
    data = np.load(args.npz, allow_pickle=False)

    # Detect time axis from joint_pos
    joint_pos = data["joint_pos"]
    num_frames = joint_pos.shape[0]
    fps = float(data.get("fps", 50.0))
    duration = num_frames / fps
    print(f"Total frames: {num_frames}, FPS: {fps}, Duration: {duration:.2f}s")

    start = max(0, args.start)
    end = min(num_frames, args.end) if args.end is not None else num_frames

    if start >= end:
        print(f"Error: start ({start}) >= end ({end})")
        sys.exit(1)

    cut_data = {}
    metadata_keys = {"fps", "joint_names", "key_body_names"}
    for key in data.files:
        arr = data[key]
        if (
            key not in metadata_keys
            and arr.ndim > 0
            and arr.shape[0] == num_frames
        ):
            cut_data[key] = arr[start:end]
            print(f"  {key}: {arr.shape} → {cut_data[key].shape}")
        else:
            cut_data[key] = arr

    new_frames = cut_data["joint_pos"].shape[0]
    new_duration = new_frames / fps
    print(f"Cut frames: [{start}, {end}) → {new_frames} frames, {new_duration:.2f}s")

    np.savez(args.out, **cut_data)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
