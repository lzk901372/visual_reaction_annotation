#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path
from typing import Iterable, List

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
            "Analyze audio emotion fluctuation by 2.4s chunks with "
            "Qwen2-Audio-7B-Instruct and save txt outputs."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="realtalk",
        help="Dataset name.",
    )
    parser.add_argument(
        "--list-file",
        # required=True,
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt"),
        help="Input txt file. Each line has three comma-separated paths; only first is used.",
    )
    parser.add_argument(
        "--input-root",
        # required=True,
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/Original_wav"),
        help="Input dataset root path used to compute relative path.",
    )
    parser.add_argument(
        "--output-root",
        # required=True,
        type=Path,
        default=Path("/data/zikai/Data/Qwen2-Audio_extract/realtalk"),
        help="Root directory to save chunk txt outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/zikai/Codes/base_checkpoints/qwen2-audio-7b-instruct"),
        help="Local path of qwen2-audio-7b-instruct model.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).resolve().parent / "input_prompt.txt",
        help="Prompt template text file path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate for loading waveform.",
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
        default=256,
        help="Max new tokens for model generation per chunk.",
    )
    parser.add_argument(
        "--chunk-index-width",
        type=int,
        default=3,
        help="Zero padding width for chunk index in filenames.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inference when output txt already exists.",
    )
    parser.add_argument(
        "--max-audios",
        type=int,
        default=None,
        help="Only process first N audios from list file for quick validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks inferred together in one forward pass.",
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
    non_null_chunks = []
    for path in output_root.glob("**/*.txt"):
        # if path.read_text(encoding="utf-8").strip() != "NULL":
        #     non_null_chunks.append(path)
        text = path.read_text(encoding="utf-8").strip()
        if "NULL" not in text:
            print(f"Non-null chunk: {path}")
            non_null_chunks.append(path)
    return non_null_chunks


def read_prompt(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8").strip()


def parse_list_file(list_file: Path, input_root: Path, dataset_name: str = "realtalk") -> Iterable[Path]:
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
                first_col = first_col.replace("Seamless", "Seamless_audio").replace("mp4", "wav")
            
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
        # For each 2.4s chunk, valid event times must be within chunk range.
        if start < 0.0 or end > max_time + 1e-6:
            continue
        if score < 0.5:
            continue
        score = min(max(score, 0.0), 1.0)
        normalized_lines.append(f"{start:.2f}, {end:.2f}, {score:.2f}")

    if not normalized_lines:
        return "NULL"
    return "\n".join(normalized_lines)


class Qwen2AudioAnalyzer:
    def __init__(self, model_path: Path, max_new_tokens: int, prompt: str) -> None:
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.prompt_text = self._build_prompt_text(prompt)

        if torch.cuda.is_available():
            # Enable TF32 matmul on supported GPUs for faster inference.
            torch.backends.cuda.matmul.allow_tf32 = True

        if torch.cuda.is_available():
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.input_device = "cuda"
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to("cpu")
            self.input_device = "cpu"

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
            conversation, add_generation_prompt=True, tokenize=False
        )

    @torch.inference_mode()
    def infer_chunks(
        self,
        chunk_waveforms: List[np.ndarray],
        sample_rate: int,
        chunk_seconds: float,
    ) -> List[str]:
        inputs = self.processor(
            text=[self.prompt_text] * len(chunk_waveforms),
            audio=chunk_waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.input_device)
        if not any(("audio" in k.lower() or "feature" in k.lower()) for k in inputs.keys()):
            raise RuntimeError(
                f"No audio features found in processor inputs. keys={list(inputs.keys())}"
            )

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
        return [normalize_result(decoded, max_time=chunk_seconds) for decoded in decoded_list]


def main() -> None:
    args = parse_args()
    if args.max_audios is not None and args.max_audios <= 0:
        raise ValueError("--max-audios must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if not (0.0 < args.peak_limit <= 1.0):
        raise ValueError("--peak-limit must be in (0, 1]")
    if args.min_chunk_rms <= 0.0:
        raise ValueError("--min-chunk-rms must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    prompt = read_prompt(args.prompt_file)
    analyzer = Qwen2AudioAnalyzer(args.model_path, args.max_new_tokens, prompt)

    total_audio = 0
    total_chunks = 0
    skipped_chunks = 0
    processed_chunk_attempts = 0
    low_rms_filtered_chunks = 0

    audio_paths = list(parse_list_file(args.list_file, args.input_root, args.dataset_name))
    if args.max_audios is not None:
        audio_paths = audio_paths[: args.max_audios]

    progress = tqdm(audio_paths, desc="Audios", unit="audio")
    start_time = time.time()

    for audio_path in progress:
        total_audio += 1

        if not audio_path.exists():
            print(f"[WARN] audio not found: {audio_path}")
            elapsed = max(time.time() - start_time, 1e-6)
            progress.set_postfix(
                chunks_inferred=total_chunks,
                chunks_skipped=skipped_chunks,
                chunk_per_sec=f"{processed_chunk_attempts / elapsed:.2f}",
            )
            continue

        try:
            waveform = load_waveform(audio_path, args.sample_rate)
            if args.normalize_loudness:
                waveform = normalize_waveform_loudness(
                    waveform,
                    target_dbfs=args.target_dbfs,
                    peak_limit=args.peak_limit,
                )
        except Exception as exc:
            print(f"[WARN] failed loading audio: {audio_path} ({exc})")
            elapsed = max(time.time() - start_time, 1e-6)
            progress.set_postfix(
                chunks_inferred=total_chunks,
                chunks_skipped=skipped_chunks,
                chunk_per_sec=f"{processed_chunk_attempts / elapsed:.2f}",
            )
            continue

        chunks = split_chunks(waveform, args.sample_rate, args.chunk_seconds)
        if not chunks:
            print(f"[INFO] no full chunk, ignore: {audio_path}")
            elapsed = max(time.time() - start_time, 1e-6)
            progress.set_postfix(
                chunks_inferred=total_chunks,
                chunks_skipped=skipped_chunks,
                chunk_per_sec=f"{processed_chunk_attempts / elapsed:.2f}",
            )
            continue

        subdir_path = to_output_subdir_path(audio_path, args.input_root)
        save_dir = args.output_root / subdir_path
        save_dir.mkdir(parents=True, exist_ok=True)

        pending_indices: List[int] = []
        pending_chunks: List[np.ndarray] = []
        for idx, chunk in enumerate(chunks):
            processed_chunk_attempts += 1
            out_path = save_dir / f"chunk_{idx:0{args.chunk_index_width}d}.txt"
            if args.skip_existing and out_path.exists():
                skipped_chunks += 1
                continue
            if args.enable_min_rms_gate and compute_rms(chunk) < args.min_chunk_rms:
                out_path.write_text("NULL\n", encoding="utf-8")
                low_rms_filtered_chunks += 1
                continue
            pending_indices.append(idx)
            pending_chunks.append(chunk)

        for start in range(0, len(pending_chunks), args.batch_size):
            batch_chunks = pending_chunks[start : start + args.batch_size]
            batch_indices = pending_indices[start : start + args.batch_size]
            try:
                batch_results = analyzer.infer_chunks(
                    batch_chunks,
                    args.sample_rate,
                    args.chunk_seconds,
                )
                for idx, result in zip(batch_indices, batch_results):
                    out_path = save_dir / f"chunk_{idx:0{args.chunk_index_width}d}.txt"
                    out_path.write_text(result + "\n", encoding="utf-8")
                    total_chunks += 1
            except Exception as exc:
                for idx in batch_indices:
                    print(f"[WARN] failed chunk infer: {audio_path} chunk={idx} ({exc})")

        elapsed = max(time.time() - start_time, 1e-6)
        progress.set_postfix(
            chunks_inferred=total_chunks,
            chunks_skipped=skipped_chunks,
            chunk_per_sec=f"{processed_chunk_attempts / elapsed:.2f}",
        )

    progress.close()

    print("===== Done =====")
    print(f"audios_seen={total_audio}")
    print(f"max_audios_limit={args.max_audios}")
    print(f"chunks_inferred={total_chunks}")
    print(f"chunks_skipped_existing={skipped_chunks}")
    print(f"chunks_low_rms_filtered={low_rms_filtered_chunks}")
    print(f"batch_size={args.batch_size}")
    print(f"normalize_loudness={args.normalize_loudness}")
    print(f"enable_min_rms_gate={args.enable_min_rms_gate}")

    if args.scan_non_null:
        print("Let's find non-null chunks...")
        find_non_null_chunks(args.output_root)


if __name__ == "__main__":
    main()
