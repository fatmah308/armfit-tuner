"""
Shared helper for locating built llama.cpp host binaries across the
different layouts CMake generators produce:
  - Ninja / Makefiles (single-config):  build/bin/<name>
  - Visual Studio (multi-config):        build/bin/Release/<name>.exe

Both quantize.py and quality.py need this exact same search, so it
lives here once instead of being duplicated in each.
"""

import platform
from pathlib import Path


def find_host_binary(llama_dir: Path, name: str) -> Path:
    exe = f"{name}.exe" if platform.system() == "Windows" else name
    candidates = [
        llama_dir / "build" / "bin" / "Release" / exe,  # Visual Studio multi-config generator
        llama_dir / "build" / "bin" / exe,               # Ninja / Makefiles single-config generator
        llama_dir / "build" / exe,
        llama_dir / exe,
    ]
    for c in candidates:
        if c.exists():
            return c
    checked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Could not find a built {name} binary under {llama_dir}. Checked:\n{checked}\n"
        "Build llama.cpp locally first (a plain host build is fine, "
        "it does not need to be the Android cross-compiled build)."
    )
