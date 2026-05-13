#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import queue
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


LINE_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU batch inference for emotion fluctuation with "
            "Qwen2-Audio-7B-Instruct."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="realtalk",
        help="Dataset name used for path rewrite rules.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt"),
        help="Input txt file. Each line has 3 comma-separated paths, only first is used.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/Original_wav"),
        help="Input dataset root path used to compute relative path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zikai/Data/Qwen2-Audio_extract/realtalk"),
        help="Root directory to save chunk txt outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/zikai/Codes/base_checkpoints/qwen2-audio-7b-instruct"),
        help="Local model directory of qwen2-audio-7b-instruct.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).resolve().parent / "input_prompt.txt",
        help="Prompt template file path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate for waveform loading.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.4,
        help="Chunk duration in seconds.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens per chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Chunk batch size per GPU worker.",
    )
    parser.add_argument(
        "--chunk-index-width",
        type=int,
        default=3,
        help="Zero-padding width for chunk index filenames.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chunks when corresponding txt already exists.",
    )
    parser.add_argument(
        "--max-audios",
        type=int,
        default=None,
        help="Only process first N audios for quick validation.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help='GPU ids for workers, e.g. "0,1,2". Default: all visible GPUs.',
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: number of provided GPU ids.",
    )
    parser.add_argument(
        "--scan-non-null",
        action="store_true",
        help="Scan output txt files at the end and print non-null chunks (slow on large outputs).",
    )
    parser.add_argument(
        "--normalize-loudness",
        action="store_true",
        help="Enable waveform loudness normalization before chunking.",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=-20.0,
        help="Target RMS loudness (dBFS) when --normalize-loudness is enabled.",
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=0.98,
        help="Peak limit after loudness normalization.",
    )
    parser.add_argument(
        "--enable-min-rms-gate",
        action="store_true",
        help="If enabled, chunks below --min-chunk-rms are directly written as NULL.",
    )
    parser.add_argument(
        "--min-chunk-rms",
        type=float,
        default=0.003,
        help="Minimum chunk RMS for inference when --enable-min-rms-gate is enabled.",
    )
    return parser.parse_args()


def find_non_null_chunks(output_root: Path) -> List[Path]:
    non_null_chunks: List[Path] = []
    for path in output_root.glob("**/*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        if "NULL" not in text:
            print(f"Non-null chunk: {path}")
            non_null_chunks.append(path)
    return non_null_chunks


def read_prompt(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8").strip()


def parse_list_file(list_file: Path, input_root: Path, dataset_name: str) -> Iterable[Path]:
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            first_col = stripped.split(",")[0].strip()
            if not first_col:
                continue

            if dataset_name == "realtalk":
                first_col = first_col.replace("Original", "Original_wav").replace("mp4", "wav")
            elif dataset_name == "seamless":
                first_col = first_col.replace("Seamless", "Seamless_audio").replace(".mp4", "_speak.wav")

            p = Path(first_col)
            if not p.is_absolute():
                p = input_root / p
            yield p


def load_waveform(audio_path: Path, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return waveform.astype(np.float32, copy=False)


def compute_rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64) + 1e-12))


def normalize_waveform_loudness(
    waveform: np.ndarray,
    target_dbfs: float,
    peak_limit: float,
) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    rms = compute_rms(waveform)
    if rms <= 0.0:
        return waveform
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / max(rms, 1e-12)
    scaled = waveform * gain
    peak = float(np.max(np.abs(scaled)) + 1e-12)
    if peak > peak_limit:
        scaled = scaled * (peak_limit / peak)
    return scaled.astype(np.float32, copy=False)


def split_chunks(waveform: np.ndarray, sample_rate: int, chunk_seconds: float) -> List[np.ndarray]:
    chunk_samples = int(round(sample_rate * chunk_seconds))
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    total_chunks = len(waveform) // chunk_samples
    return [
        waveform[i * chunk_samples : (i + 1) * chunk_samples]
        for i in range(total_chunks)
    ]


def to_output_subdir_path(audio_path: Path, input_root: Path) -> Path:
    rel_path = Path(os.path.relpath(audio_path, input_root))
    return rel_path.with_suffix("")


def normalize_result(text: str, max_time: float) -> str:
    raw = text.strip()
    if not raw:
        return "NULL"

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    normalized_lines: List[str] = []
    for ln in lines:
        if ln.upper() == "NULL":
            continue
        m = LINE_PATTERN.match(ln)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        score = float(m.group(3))
        if end <= start:
            continue
        if start < 0.0 or end > max_time + 1e-6:
            continue
        if score < 0.5:
            continue
        score = min(max(score, 0.0), 1.0)
        normalized_lines.append(f"{start:.2f}, {end:.2f}, {score:.2f}")

    if not normalized_lines:
        return "NULL"
    return "\n".join(normalized_lines)


def partition_list(items: Sequence[Path], num_parts: int) -> List[List[Path]]:
    parts: List[List[Path]] = [[] for _ in range(num_parts)]
    for i, item in enumerate(items):
        parts[i % num_parts].append(item)
    return parts


class Qwen2AudioAnalyzer:
    def __init__(self, model_path: Path, prompt: str, max_new_tokens: int, device: str) -> None:
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.prompt_text = self._build_prompt_text(prompt)

        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device)
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(device)
        self.model.eval()

    def _build_prompt_text(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "chunk.wav"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    @staticmethod
    def _has_audio_features(inputs: Dict[str, Any]) -> bool:
        keys = [k.lower() for k in inputs.keys()]
        return any(
            ("audio" in k)
            or ("feature" in k)
            or (k in {"input_values", "input_features"})
            for k in keys
        )

    def _to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def _build_model_inputs(
        self, chunk_waveforms: List[np.ndarray], sample_rate: int
    ) -> Dict[str, Any]:
        text_batch = [self.prompt_text] * len(chunk_waveforms)
        errors: List[str] = []

        for audio_kw in ("audios", "audio"):
            try:
                cand = self.processor(
                    text=text_batch,
                    # **{audio_kw: chunk_waveforms},
                    audio=chunk_waveforms,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                cand = self._to_device(dict(cand))
                if self._has_audio_features(cand):
                    return cand
                errors.append(f"processor kw={audio_kw} missing audio keys={list(cand.keys())}")
            except Exception as exc:
                errors.append(f"processor kw={audio_kw} failed={exc}")

        try:
            text_inputs = self.processor.tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
            )
            audio_inputs = self.processor.feature_extractor(
                chunk_waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            merged = {**dict(text_inputs), **dict(audio_inputs)}
            merged = self._to_device(merged)
            if self._has_audio_features(merged):
                return merged
            errors.append(f"fallback missing audio keys={list(merged.keys())}")
        except Exception as exc:
            errors.append(f"fallback failed={exc}")

        raise RuntimeError("No audio features found. " + " | ".join(errors))

    @torch.inference_mode()
    def infer_chunks(
        self,
        chunk_waveforms: List[np.ndarray],
        sample_rate: int,
        chunk_seconds: float,
    ) -> List[str]:
        inputs = self._build_model_inputs(chunk_waveforms, sample_rate)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].size(1) :]
        decoded_list = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [normalize_result(x, max_time=chunk_seconds) for x in decoded_list]


def worker_main(
    worker_idx: int,
    gpu_id: int,
    audio_paths: List[Path],
    args_dict: Dict[str, Any],
    progress_q: mp.Queue,
) -> None:
    try:
        if gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

        analyzer = Qwen2AudioAnalyzer(
            model_path=Path(args_dict["model_path"]),
            prompt=args_dict["prompt"],
            max_new_tokens=args_dict["max_new_tokens"],
            device=device,
        )

        for audio_path in audio_paths:
            delta_inferred = 0
            delta_skipped = 0
            delta_attempts = 0
            delta_low_rms_filtered = 0

            if not audio_path.exists():
                progress_q.put(
                    {
                        "type": "progress",
                        "audios_done": 1,
                        "chunks_inferred": 0,
                        "chunks_skipped": 0,
                        "chunk_attempts": 0,
                        "warn": f"[W{worker_idx}] audio not found: {audio_path}",
                    }
                )
                continue

            try:
                waveform = load_waveform(audio_path, args_dict["sample_rate"])
                if args_dict["normalize_loudness"]:
                    waveform = normalize_waveform_loudness(
                        waveform,
                        target_dbfs=args_dict["target_dbfs"],
                        peak_limit=args_dict["peak_limit"],
                    )
            except Exception as exc:
                progress_q.put(
                    {
                        "type": "progress",
                        "audios_done": 1,
                        "chunks_inferred": 0,
                        "chunks_skipped": 0,
                        "chunk_attempts": 0,
                        "warn": f"[W{worker_idx}] failed loading audio: {audio_path} ({exc})",
                    }
                )
                continue

            chunks = split_chunks(waveform, args_dict["sample_rate"], args_dict["chunk_seconds"])
            if not chunks:
                progress_q.put(
                    {
                        "type": "progress",
                        "audios_done": 1,
                        "chunks_inferred": 0,
                        "chunks_skipped": 0,
                        "chunk_attempts": 0,
                        "warn": f"[W{worker_idx}] no full chunk, ignore: {audio_path}",
                    }
                )
                continue

            subdir_path = to_output_subdir_path(audio_path, Path(args_dict["input_root"]))
            save_dir = Path(args_dict["output_root"]) / subdir_path
            save_dir.mkdir(parents=True, exist_ok=True)

            pending_indices: List[int] = []
            pending_chunks: List[np.ndarray] = []
            for idx, chunk in enumerate(chunks):
                delta_attempts += 1
                out_path = save_dir / f"chunk_{idx:0{args_dict['chunk_index_width']}d}.txt"
                if args_dict["skip_existing"] and out_path.exists():
                    delta_skipped += 1
                    continue
                if (
                    args_dict["enable_min_rms_gate"]
                    and compute_rms(chunk) < args_dict["min_chunk_rms"]
                ):
                    out_path.write_text("NULL\n", encoding="utf-8")
                    delta_low_rms_filtered += 1
                    continue
                pending_indices.append(idx)
                pending_chunks.append(chunk)

            for start in range(0, len(pending_chunks), args_dict["batch_size"]):
                batch_chunks = pending_chunks[start : start + args_dict["batch_size"]]
                batch_indices = pending_indices[start : start + args_dict["batch_size"]]
                try:
                    batch_results = analyzer.infer_chunks(
                        batch_chunks,
                        args_dict["sample_rate"],
                        args_dict["chunk_seconds"],
                    )
                    for idx, result in zip(batch_indices, batch_results):
                        out_path = save_dir / f"chunk_{idx:0{args_dict['chunk_index_width']}d}.txt"
                        out_path.write_text(result + "\n", encoding="utf-8")
                        delta_inferred += 1
                except Exception as exc:
                    for idx in batch_indices:
                        progress_q.put(
                            {
                                "type": "log",
                                "warn": (
                                    f"[W{worker_idx}] failed chunk infer: {audio_path} "
                                    f"chunk={idx} ({exc})"
                                ),
                            }
                        )

            progress_q.put(
                {
                    "type": "progress",
                    "audios_done": 1,
                    "chunks_inferred": delta_inferred,
                    "chunks_skipped": delta_skipped,
                    "chunk_attempts": delta_attempts,
                    "chunks_low_rms_filtered": delta_low_rms_filtered,
                }
            )

        progress_q.put({"type": "done", "worker_idx": worker_idx})
    except Exception as exc:
        progress_q.put({"type": "log", "warn": f"[W{worker_idx}] fatal error: {exc}"})
        progress_q.put({"type": "done", "worker_idx": worker_idx})


def parse_gpu_ids(gpu_ids_arg: str | None) -> List[int]:
    if gpu_ids_arg is None:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [-1]
    ids: List[int] = []
    for x in gpu_ids_arg.split(","):
        x = x.strip()
        if not x:
            continue
        ids.append(int(x))
    if not ids:
        raise ValueError("--gpu-ids is empty")
    return ids


def main() -> None:
    args = parse_args()
    if args.max_audios is not None and args.max_audios <= 0:
        raise ValueError("--max-audios must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not (0.0 < args.peak_limit <= 1.0):
        raise ValueError("--peak-limit must be in (0, 1]")
    if args.min_chunk_rms <= 0.0:
        raise ValueError("--min-chunk-rms must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    prompt = read_prompt(args.prompt_file)

    audio_paths = list(parse_list_file(args.list_file, args.input_root, args.dataset_name))
    if args.max_audios is not None:
        audio_paths = audio_paths[: args.max_audios]
    if not audio_paths:
        print("No audio paths found.")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.num_workers is None:
        num_workers = len(gpu_ids)
    else:
        num_workers = args.num_workers
    if num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if num_workers > len(gpu_ids):
        raise ValueError("--num-workers cannot exceed number of gpu ids")

    gpu_ids = gpu_ids[:num_workers]
    shards = partition_list(audio_paths, num_workers)

    print("===== Launch Config =====")
    print(f"num_audios={len(audio_paths)}")
    print(f"gpu_ids={gpu_ids}")
    print(f"num_workers={num_workers}")
    print(f"batch_size={args.batch_size}")
    print(f"max_new_tokens={args.max_new_tokens}")

    ctx = mp.get_context("spawn")
    progress_q: mp.Queue = ctx.Queue()
    workers: List[mp.Process] = []

    args_dict = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "model_path": str(args.model_path),
        "prompt": prompt,
        "sample_rate": args.sample_rate,
        "chunk_seconds": args.chunk_seconds,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "chunk_index_width": args.chunk_index_width,
        "skip_existing": args.skip_existing,
        "normalize_loudness": args.normalize_loudness,
        "target_dbfs": args.target_dbfs,
        "peak_limit": args.peak_limit,
        "enable_min_rms_gate": args.enable_min_rms_gate,
        "min_chunk_rms": args.min_chunk_rms,
    }

    for worker_idx, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=worker_main,
            args=(worker_idx, gpu_id, shards[worker_idx], args_dict, progress_q),
            daemon=False,
        )
        p.start()
        workers.append(p)

    total_audio = 0
    total_chunks = 0
    skipped_chunks = 0
    chunk_attempts = 0
    low_rms_filtered_chunks = 0
    done_workers = 0
    start_time = time.time()

    progress = tqdm(total=len(audio_paths), desc="Audios", unit="audio")
    while done_workers < num_workers:
        try:
            msg = progress_q.get(timeout=1.0)
        except queue.Empty:
            continue

        msg_type = msg.get("type")
        if msg_type == "progress":
            progress.update(int(msg.get("audios_done", 0)))
            total_audio += int(msg.get("audios_done", 0))
            total_chunks += int(msg.get("chunks_inferred", 0))
            skipped_chunks += int(msg.get("chunks_skipped", 0))
            chunk_attempts += int(msg.get("chunk_attempts", 0))
            low_rms_filtered_chunks += int(msg.get("chunks_low_rms_filtered", 0))
            warn = msg.get("warn")
            if warn:
                print(warn)

            elapsed = max(time.time() - start_time, 1e-6)
            progress.set_postfix(
                chunks_inferred=total_chunks,
                chunks_skipped=skipped_chunks,
                chunks_low_rms=low_rms_filtered_chunks,
                chunk_per_sec=f"{chunk_attempts / elapsed:.2f}",
            )
        elif msg_type == "log":
            warn = msg.get("warn")
            if warn:
                print(warn)
        elif msg_type == "done":
            done_workers += 1

    progress.close()

    for p in workers:
        p.join()

    print("===== Done =====")
    print(f"audios_seen={total_audio}")
    print(f"chunks_inferred={total_chunks}")
    print(f"chunks_skipped_existing={skipped_chunks}")
    print(f"chunks_low_rms_filtered={low_rms_filtered_chunks}")
    print(f"num_workers={num_workers}")
    print(f"gpu_ids={gpu_ids}")
    print(f"normalize_loudness={args.normalize_loudness}")
    print(f"enable_min_rms_gate={args.enable_min_rms_gate}")

    if args.scan_non_null:
        print("Let's find non-null chunks...")
        find_non_null_chunks(args.output_root)


if __name__ == "__main__":
    main()
