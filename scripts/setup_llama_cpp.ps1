<#
.SYNOPSIS
    Builds TWO Android arm64-v8a binaries of llama.cpp:
      1. build-baseline\  - generic build, no Arm-specific matmul kernels
      2. build-kleidiai\  - GGML_CPU_KLEIDIAI=ON, targets DotProd (and I8MM
         if you edit ARCH_FLAGS below and your device supports it)

    This is the core "Arm-specific optimization" comparison: the same
    model, same quant level, same device, two different builds of the
    same framework, with Arm's own accelerated kernels toggled on or off.

    NOTE: cmake arguments are passed via PowerShell array splatting
    (@array) rather than backtick line-continuation. Backtick continuation
    is fragile, a stray line-wrap from copy-pasting or manual editing can
    silently truncate an argument with no error, which is exactly what
    happened during development of this script (the arch flag got cut
    from "armv8.2-a+dotprod" down to just "armv8"). Splatting passes each
    array element as one atomic argument, so this class of bug can't
    happen regardless of how the file gets edited or pasted later.

.REQUIREMENTS
    - Android NDK r27 or later
    - cmake and ninja, on PATH

.USAGE
    $env:NDK_PATH = "C:\Users\<you>\AppData\Local\Android\Sdk\ndk\27.0.12077973"
    .\setup_llama_cpp.ps1 -LlamaDir D:\armfit-tuner\llama.cpp
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$LlamaDir,

    [string]$NdkPath = $env:NDK_PATH
)

# Edit this if your device supports I8MM (check with: adb shell cat /proc/cpuinfo).
# Kept as a single named variable, not inline in a long flag string, specifically
# so it can't get silently truncated by a line-wrap the way it did before.
$ArchFlags = "armv8.2-a+dotprod"

if (-not $NdkPath) {
    Write-Error "NDK_PATH not set. Either run `$env:NDK_PATH = '<path>' first, or pass -NdkPath explicitly."
    exit 1
}

$Toolchain = Join-Path $NdkPath "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $Toolchain)) {
    Write-Error "Could not find NDK toolchain file at $Toolchain`nCheck that NdkPath points at the NDK root (the folder containing build\cmake\...)."
    exit 1
}

if (-not (Test-Path $LlamaDir)) {
    Write-Error "LlamaDir '$LlamaDir' does not exist."
    exit 1
}

$Jobs = $env:NUMBER_OF_PROCESSORS
if (-not $Jobs) { $Jobs = 4 }

if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "Ninja is required for cross-compiling to Android on Windows (the default Visual Studio generator will not produce correct Android binaries). Install it with: pip install ninja --break-system-packages`nThen make sure the pip scripts folder is on PATH."
    exit 1
}

Push-Location $LlamaDir
try {
    $commonConfigureArgs = @(
        "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-24",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_BUILD_WEBUI=OFF"
    )

    Write-Host "==> Building BASELINE (no Arm-specific matmul kernels) ..." -ForegroundColor Cyan
    $baselineConfigureArgs = @("-B", "build-baseline", "-G", "Ninja") + $commonConfigureArgs + @("-DGGML_CPU_KLEIDIAI=OFF")
    Write-Host "    cmake args: $($baselineConfigureArgs -join ' ')"
    & cmake @baselineConfigureArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (baseline) failed with exit code $LASTEXITCODE" }

    & cmake --build build-baseline --config Release --target llama-bench -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "cmake build (baseline) failed with exit code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "==> Building KLEIDIAI (arch flags: $ArchFlags) ..." -ForegroundColor Cyan
    Write-Host "    Note: if your device's CPU lacks a feature in these flags, the"
    Write-Host "    binary can crash at runtime (illegal instruction). Check actual"
    Write-Host "    support on your phone first with:"
    Write-Host "        adb shell cat /proc/cpuinfo" -ForegroundColor Yellow
    $kleidiaiConfigureArgs = @("-B", "build-kleidiai", "-G", "Ninja") + $commonConfigureArgs + @(
        "-DGGML_CPU_KLEIDIAI=ON",
        "-DGGML_CPU_ARM_ARCH=$ArchFlags"
    )
    Write-Host "    cmake args: $($kleidiaiConfigureArgs -join ' ')"
    & cmake @kleidiaiConfigureArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (kleidiai) failed with exit code $LASTEXITCODE" }

    & cmake --build build-kleidiai --config Release --target llama-bench -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "cmake build (kleidiai) failed with exit code $LASTEXITCODE" }

    # llama.cpp's CPU backend links against LLVM's OpenMP runtime (libomp.so),
    # which lives inside the NDK toolchain itself, not in llama.cpp's own build
    # output. armfit's device-push step only auto-discovers .so files sitting
    # next to the binary, so copy libomp.so there now, once, for both builds.
    Write-Host ""
    Write-Host "==> Locating libomp.so (OpenMP runtime) in the NDK ..." -ForegroundColor Cyan
    $libomp = Get-ChildItem -Path $NdkPath -Recurse -Filter "libomp.so" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*aarch64*" } | Select-Object -First 1
    if ($libomp) {
        Write-Host "    Found: $($libomp.FullName)"
        Copy-Item $libomp.FullName -Destination (Join-Path $LlamaDir "build-baseline\bin\") -Force
        Copy-Item $libomp.FullName -Destination (Join-Path $LlamaDir "build-kleidiai\bin\") -Force
        Write-Host "    Copied into both build-baseline\bin and build-kleidiai\bin"
    } else {
        Write-Warning "Could not find libomp.so (aarch64) under the NDK. If the on-device run later fails with a missing libomp.so error, locate it manually and copy it into build-baseline\bin and build-kleidiai\bin."
    }

    Write-Host ""
    Write-Host "==> Done." -ForegroundColor Green
    Write-Host "Baseline binary:  $LlamaDir\build-baseline\bin\llama-bench"
    Write-Host "KleidiAI binary:  $LlamaDir\build-kleidiai\bin\llama-bench"
    Write-Host ""
    Write-Host "Pass these two paths to armfit via --baseline-bin and --kleidiai-bin"
}
finally {
    Pop-Location
}