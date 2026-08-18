"""Cut PKL motion data by frame range.

Usage:
    # Cut frames 100 to 500
    python tools/cut_pkl.py --pkl input.pkl --out output.pkl --start 100 --end 500

    # Cut first 200 frames (from beginning)
    python tools/cut_pkl.py --pkl input.pkl --out output.pkl --end 200

    # Cut from frame 300 to end
    python tools/cut_pkl.py --pkl input.pkl --out output.pkl --start 300
"""
import argparse
import io
import pickle
import sys

class NumpyCompatUnpickler(pickle.Unpickler):
    """Handle NumPy 2.x namespace changes during unpickling."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pkl(path: str) -> dict:
    """Load a pickle file with NumPy 2.x compatibility."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return pickle.loads(raw)
    except (AttributeError, ModuleNotFoundError, SystemError, TypeError):
        return NumpyCompatUnpickler(io.BytesIO(raw)).load()


def main():
    parser = argparse.ArgumentParser(description="Cut PKL motion data by frame range")
    parser.add_argument("--pkl", type=str, required=True, help="Input PKL file")
    parser.add_argument("--out", type=str, required=True, help="Output PKL file")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive, 0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End frame (exclusive, 0-indexed)")
    args = parser.parse_args()

    print(f"Loading: {args.pkl}")
    data = load_pkl(args.pkl)

    num_frames = data["root_pos"].shape[0]
    fps = data.get("fps", 50.0)
    duration = num_frames / fps

    print(f"Total frames: {num_frames}, FPS: {fps}, Duration: {duration:.2f}s")

    start = max(0, args.start)
    end = min(num_frames, args.end) if args.end is not None else num_frames

    if start >= end:
        print(f"Error: start ({start}) >= end ({end})")
        sys.exit(1)

    # Validate
    if start >= num_frames:
        print(f"Error: start ({start}) >= total frames ({num_frames})")
        sys.exit(1)

    # Slice all time-dimensioned arrays
    cut_data = {}
    time_keys = ["root_pos", "root_rot", "dof_pos"]
    optional_time_keys = ["root_vel_body", "root_rot_vel", "dof_vel"]

    for key in time_keys:
        if key in data:
            arr = data[key]
            cut_data[key] = arr[start:end]
            print(f"  {key}: {arr.shape} → {cut_data[key].shape}")

    for key in optional_time_keys:
        if key in data:
            arr = data[key]
            cut_data[key] = arr[start:end]
            print(f"  {key}: {arr.shape} → {cut_data[key].shape}")

    # Copy scalar/metadata keys
    for key, val in data.items():
        if isinstance(val, (int, float, str, bool, type(None))):
            cut_data[key] = val

    # Force fps (it might be a scalar, ensure it's copied)
    if "fps" in data and "fps" not in cut_data:
        cut_data["fps"] = data["fps"]

    new_frames = cut_data["root_pos"].shape[0]
    new_duration = new_frames / fps
    print(f"Cut frames: [{start}, {end}) → {new_frames} frames, {new_duration:.2f}s")

    with open(args.out, "wb") as f:
        pickle.dump(cut_data, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
