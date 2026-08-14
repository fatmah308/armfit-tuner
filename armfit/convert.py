"""
Download a Hugging Face model and convert it to GGUF (F16) using
llama.cpp's own convert_hf_to_gguf.py script.
"""

import subprocess
import sys
from pathlib import Path


class ConvertError(RuntimeError):
    pass


def hf_to_gguf(model_id: str, llama_dir: Path, work_dir: Path) -> Path:
    """
    Downloads `model_id` from the Hugging Face Hub and converts it to
    an F16 GGUF file inside work_dir. Returns the path to the .gguf file.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ConvertError(
            "huggingface_hub is required. Install with: pip install huggingface_hub --break-system-packages"
        ) from e

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise ConvertError(
            f"Could not find convert_hf_to_gguf.py in {llama_dir}. "
            "Point --llama-dir at a full clone of https://github.com/ggml-org/llama.cpp"
        )

    model_dir = work_dir / "hf_model" / model_id.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"      Downloading {model_id} ...")
    snapshot_download(repo_id=model_id, local_dir=model_dir)

    out_gguf = work_dir / f"{model_id.replace('/', '__')}-f16.gguf"

    cmd = [
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(out_gguf),
        "--outtype", "f16",
    ]
    print(f"      Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise ConvertError(f"convert_hf_to_gguf.py failed:\n{result.stderr}")

    if not out_gguf.exists():
        raise ConvertError(f"Conversion reported success but {out_gguf} was not created.")

    return out_gguf
