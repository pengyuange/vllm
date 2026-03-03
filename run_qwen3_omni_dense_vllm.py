"""
vLLM offline inference demo for Qwen3-Omni (dense) model.

Mirrors the usage in run_qwen3_omni_dense.py but uses vLLM for inference.
Supports text-only, audio-only, image-only, and combined (image + audio) inputs.
"""

import io
import urllib.request

import numpy as np
import soundfile as sf
from PIL import Image

from vllm import LLM, SamplingParams

MODEL_PATH = "/Users/gepengyuan/Downloads/Qwen3-Omni-4B-Instruct-multilingual"

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text."
)

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def build_prompt(user_content: str, system: str = SYSTEM_PROMPT) -> str:
    """Build a Qwen-style chat prompt with placeholders already embedded."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Media loaders
# ---------------------------------------------------------------------------

def load_audio_from_url(url: str) -> tuple[np.ndarray, int]:
    """Download audio from URL and return (samples, sample_rate)."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return audio, sr


def load_audio_from_file(path: str) -> tuple[np.ndarray, int]:
    """Load audio from a local file and return (samples, sample_rate)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return audio, sr


def load_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return a PIL Image."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    """Load image from a local file and return a PIL Image."""
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Inference examples
# ---------------------------------------------------------------------------

def run_text_only(llm: LLM, sampling_params: SamplingParams) -> None:
    print("\n" + "=" * 60)
    print("Example 1: Text-only")
    print("=" * 60)

    prompt = build_prompt("What is the capital of France?")
    outputs = llm.generate(
        [{"prompt": prompt}],
        sampling_params=sampling_params,
    )
    print("Answer:", outputs[0].outputs[0].text.strip())


def run_audio_only(llm: LLM, sampling_params: SamplingParams) -> None:
    print("\n" + "=" * 60)
    print("Example 2: Audio input")
    print("=" * 60)

    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
    print(f"Loading audio from: {audio_url}")
    audio, sr = load_audio_from_url(audio_url)
    print(f"  -> {len(audio) / sr:.2f}s @ {sr}Hz")

    prompt = build_prompt(f"{AUDIO_PLACEHOLDER}\nWhat sound do you hear? Describe it briefly.")
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": [(audio, sr)]},
            }
        ],
        sampling_params=sampling_params,
    )
    print("Answer:", outputs[0].outputs[0].text.strip())


def run_image_only(llm: LLM, sampling_params: SamplingParams) -> None:
    print("\n" + "=" * 60)
    print("Example 3: Image input")
    print("=" * 60)

    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
    print(f"Loading image from: {image_url}")
    image = load_image_from_url(image_url)
    print(f"  -> {image.size[0]}x{image.size[1]} RGB")

    prompt = build_prompt(f"{IMAGE_PLACEHOLDER}\nWhat do you see in this image? Answer in one short sentence.")
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": [image]},
            }
        ],
        sampling_params=sampling_params,
    )
    print("Answer:", outputs[0].outputs[0].text.strip())


def run_audio_and_image(llm: LLM, sampling_params: SamplingParams) -> None:
    print("\n" + "=" * 60)
    print("Example 4: Audio + Image (mirrors run_qwen3_omni_dense.py)")
    print("=" * 60)

    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"

    print(f"Loading image from: {image_url}")
    image = load_image_from_url(image_url)

    print(f"Loading audio from: {audio_url}")
    audio, sr = load_audio_from_url(audio_url)

    question = "What can you see and hear? Answer in one short sentence."
    prompt = build_prompt(
        f"{IMAGE_PLACEHOLDER}{AUDIO_PLACEHOLDER}\n{question}"
    )

    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [image],
                    "audio": [(audio, sr)],
                },
            }
        ],
        sampling_params=sampling_params,
    )
    print("Answer:", outputs[0].outputs[0].text.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model from: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=8192,
        max_num_seqs=4,
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 0},
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.0,   # greedy for reproducibility
        max_tokens=256,
        repetition_penalty=1.05,
    )

    run_text_only(llm, sampling_params)
    run_audio_only(llm, sampling_params)
    run_image_only(llm, sampling_params)
    run_audio_and_image(llm, sampling_params)


if __name__ == "__main__":
    main()
