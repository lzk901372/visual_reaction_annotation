import argparse
import pickle
from pathlib import Path


def summarize_frame(frame):
    if frame is None:
        return {"type": "None"}
    if not isinstance(frame, dict):
        return {"type": str(type(frame))}

    landmarks = frame.get("landmarks")
    blendshapes = frame.get("blendshapes", {})
    top_blend = sorted(
        [(k, float(v)) for k, v in blendshapes.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "detected": frame.get("detected"),
        "landmarks_count": len(landmarks) if isinstance(landmarks, list) else 0,
        "first_landmark": landmarks[0] if isinstance(landmarks, list) and landmarks else None,
        "bbox_norm": frame.get("bbox_norm"),
        "bbox_px": frame.get("bbox_px"),
        "au04_proxy": frame.get("au04_proxy"),
        "pitch": frame.get("pitch"),
        "roll": frame.get("roll"),
        "yaw": frame.get("yaw"),
        "top5_blendshapes": top_blend,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect face_detection pkl output.")
    parser.add_argument(
        "--pkl_path", 
        default="./output_seamless/V00_S0439_I00000538_P0006A_33.281_41.261.pkl",
        help="Path to .pkl file.",
    )
    parser.add_argument(
        "--sample_indices",
        default="0,1,2,-1",
        help="Comma-separated frame indices to preview, e.g. 0,10,100,-1",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    print(f"File: {pkl_path}")
    print(f"Top-level type: {type(data)}")

    if not isinstance(data, list):
        print("Top-level object is not a list. Printing raw object:")
        print(data)
        return

    total = len(data)
    detected = sum(
        1
        for x in data
        if isinstance(x, dict) and (x.get("detected") is True or x.get("landmarks") is not None)
    )
    print(f"Total frames: {total}")
    print(f"Detected frames: {detected}")
    if total > 0:
        print(f"Detection rate: {detected / total:.4f}")

    key_union = set()
    for x in data:
        if isinstance(x, dict):
            key_union.update(x.keys())
    print(f"Observed frame keys: {sorted(key_union)}")

    raw_indices = [s.strip() for s in args.sample_indices.split(",") if s.strip()]
    print("\nSample frames:")
    for s in raw_indices:
        idx = int(s)
        if idx < 0:
            idx = total + idx
        if idx < 0 or idx >= total:
            print(f"- frame[{s}] -> out of range")
            continue
        print(f"- frame[{idx}] -> {summarize_frame(data[idx])}")


if __name__ == "__main__":
    main()
