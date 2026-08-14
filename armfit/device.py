"""
Thin wrapper around adb for pushing binaries/models and running
benchmarks on a connected Arm Android device (tested against a
Samsung Galaxy A16, 8 GB RAM).
"""

import re
import subprocess
from pathlib import Path


class DeviceError(RuntimeError):
    pass


class AdbDevice:
    def __init__(self, serial: str, remote_dir: str):
        self.serial = serial
        self.remote_dir = remote_dir

    def _adb(self, *args, check=True):
        cmd = ["adb", "-s", self.serial, *args]
        # encoding="utf-8" is explicit here on purpose: Windows' default
        # subprocess text decoding is NOT UTF-8 (usually cp1252/mbcs), and
        # llama-bench prints the ± character as UTF-8. Without forcing this,
        # "±" gets mangled into "Â±", which breaks the results-table parser
        # downstream even though the benchmark itself ran successfully.
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if check and result.returncode != 0:
            raise DeviceError(f"adb command failed: {' '.join(cmd)}\n{result.stderr}")
        return result

    def ensure_ready(self):
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if self.serial not in result.stdout:
            raise DeviceError(
                f"Device '{self.serial}' not found in `adb devices`. "
                "Enable USB debugging (Settings > Developer options) and authorize this computer."
            )
        self._adb("shell", "mkdir", "-p", self.remote_dir)

    def free_ram_mb(self) -> int:
        result = self._adb("shell", "cat", "/proc/meminfo")
        match = re.search(r"MemAvailable:\s+(\d+)\s+kB", result.stdout)
        if not match:
            # Fall back to MemFree if MemAvailable isn't reported
            match = re.search(r"MemFree:\s+(\d+)\s+kB", result.stdout)
        if not match:
            raise DeviceError("Could not parse /proc/meminfo for free RAM.")
        return int(match.group(1)) // 1024

    def push_model(self, local_gguf: Path) -> str:
        remote_path = f"{self.remote_dir}/{local_gguf.name}"
        self._adb("push", str(local_gguf), remote_path)
        return remote_path

    def push_build(self, bin_local_path: Path, label: str) -> str:
        """
        Pushes a binary AND all shared libraries sitting alongside it.

        llama.cpp's CMake build produces llama-bench as a thin executable
        dynamically linked against separate .so files (ggml backend libs,
        libllama, etc.), so pushing the executable alone is not enough,
        the device linker needs the .so files too.

        Each build variant (baseline / kleidiai) gets its OWN remote
        subdirectory, because both variants produce .so files with the
        SAME filenames but DIFFERENT content (that's the entire point of
        the comparison). Pushing both sets into one shared folder would
        silently let the second push overwrite the first build's
        libraries, corrupting the comparison without any visible error.

        Note: some dependencies (e.g. libomp.so, LLVM's OpenMP runtime)
        live inside the NDK toolchain rather than next to the binary. The
        build script (setup_llama_cpp.ps1 / .sh) copies libomp.so into
        each build's local bin/ folder after compiling specifically so it
        gets picked up by the glob below automatically.

        Returns the remote path to the pushed binary.
        """
        remote_build_dir = f"{self.remote_dir}/{label}"
        self._adb("shell", "mkdir", "-p", remote_build_dir)

        bin_dir = bin_local_path.parent
        so_files = list(bin_dir.glob("*.so"))

        remote_bin_path = f"{remote_build_dir}/{bin_local_path.name}"
        self._adb("push", str(bin_local_path), remote_bin_path)
        self._adb("shell", "chmod", "755", remote_bin_path)

        for so_file in so_files:
            remote_so_path = f"{remote_build_dir}/{so_file.name}"
            self._adb("push", str(so_file), remote_so_path)

        return remote_bin_path

    def shell(self, command: str) -> str:
        result = self._adb("shell", command)
        return result.stdout