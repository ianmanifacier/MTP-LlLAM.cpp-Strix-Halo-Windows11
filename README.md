# MTP-LLAMA.cpp-Strix-Halo-Windows11
This guide builds llama.cpp from source on Windows 11 with HIP/ROCm support for the AMD Radeon 8060S iGPU (gfx1151) found in AMD Ryzen AI Max+ 395 ("Strix Halo"). This is just instructions and command line. It worked for me, I hope it will work for you. I used Claude Opus 4.7 to debug various steps. Gemini 3.1 pro, wasn't helpfull.

# llama.cpp with MTP Speculative Decoding on Strix Halo (Windows 11)

This guide builds llama.cpp from source on Windows 11 with HIP/ROCm support for the AMD Radeon 8060S iGPU (`gfx1151`) found in AMD Ryzen AI Max+ 395 ("Strix Halo") systems such as the Beelink GTR9 Pro. It then runs Qwen3.6-27B with Multi-Token Prediction (MTP) speculative decoding.

**Why this build instead of LM Studio:** LM Studio works fine for plain inference but does not expose llama.cpp's MTP speculative decoding flags. The only reason to go through this multi-step build is to use MTP. If you only need plain inference, install LM Studio and stop here.

**Expected outcome:** A working `llama-server.exe` that loads Qwen3.6-27B onto the iGPU and serves OpenAI-compatible chat completions with ~1.2–1.75× speedup from MTP vs the same model without it. Measured on Strix Halo: ~11.9 tok/s baseline, ~15–20 tok/s with MTP (depending on context length).

---

## Hardware and OS prerequisites

- **CPU/iGPU:** AMD Ryzen AI Max+ 395 with Radeon 8060S Graphics (`gfx1151`)
- **RAM:** 64 GB minimum; 128 GB recommended (the iGPU shares system RAM)
- **OS:** Windows 11
- **Disk:** ~40 GB free (toolchain ~5 GB, llama.cpp build ~2 GB, model ~18 GB, working space)
- **iGPU memory allocation:** In BIOS, set the GPU memory split to give the iGPU at least 96 GB. The system used in this writeup was configured for ~110 GB iGPU memory.

---

## Step 1: Install Visual Studio 2022 Build Tools

The AMD HIP clang headers ship with assumptions about MSVC compiler internals. Versions of MSVC newer than 14.4x (i.e., the 14.5x range shipped with VS 2026 Build Tools) currently conflict with HIP headers. You need MSVC 14.4x specifically, which comes with the VS 2022 Build Tools.

1. Download the VS 2022 Build Tools bootstrapper from <https://aka.ms/vs/17/release/vs_BuildTools.exe>
2. Run the installer. Select the **"Desktop development with C++"** workload.
3. Make sure the following individual components are checked:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
   - Windows 11 SDK (any recent version, e.g., 10.0.26100.0)
   - C++ CMake tools for Windows
4. Install. This places MSVC under `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.xxxxx`.

Important: from this point on, **always use the "x64 Native Tools Command Prompt for VS 2022"** to launch your build shell. This sets `INCLUDE`, `LIB`, and `PATH` to the right MSVC. The 2026 Native Tools prompt will pull in 14.5x and break the build.

If you do not have VS Build Tools 2026 installed, the plain `cmd.exe` or `powershell.exe` will not have the MSVC environment set. The VS 2022 Native Tools Command Prompt is what you want.

---

## Step 2: Install Git, CMake, and Ninja

If you do not have these:

- **Git for Windows:** <https://git-scm.com/download/win>
- **CMake:** <https://cmake.org/download/> — add to PATH during install.
- **Ninja:** download `ninja-win.zip` from <https://github.com/ninja-build/ninja/releases>, extract `ninja.exe` to a folder on PATH (e.g., `C:\tools\ninja\`).

Verify in a fresh terminal:
```powershell
git --version
cmake --version
ninja --version
```

---

## Step 3: Install TheRock ROCm toolchain (gfx1151 Windows build)

AMD's official ROCm for Windows does not yet ship a `gfx1151` build. TheRock project provides nightly builds that do.

1. Open <https://github.com/ROCm/TheRock/releases> in a browser.
2. Find the most recent **Windows gfx1151** release (look for filenames containing `windows` and `gfx1151`).
3. Download the archive (usually a `.tar.xz` or `.zip`, ~3 GB).
4. Extract to `C:\therock`. After extraction you should see:
   - `C:\therock\bin\hipcc.exe`
   - `C:\therock\bin\amdhip64_7.dll`
   - `C:\therock\lib\llvm\bin\clang.exe`

`rocminfo.exe` is **not** included in Windows TheRock builds. This is normal. Device enumeration still works through `llama-server.exe --list-devices` later.

---

## Step 4: Set HIP environment variables

In every PowerShell session where you will build or run llama.cpp, set these:

```powershell
$env:HIP_PATH      = "C:\therock"
$env:HIP_PATH_70   = "C:\therock"
$env:LLVM_PATH     = "C:\therock"
$env:HIP_PLATFORM  = "amd"
$env:PATH          = "C:\therock\bin;C:\therock\lib\llvm\bin;$env:PATH"
```

To make these permanent for your user (recommended once the build is verified working):

```powershell
[Environment]::SetEnvironmentVariable("HIP_PATH",     "C:\therock", "User")
[Environment]::SetEnvironmentVariable("HIP_PATH_70",  "C:\therock", "User")
[Environment]::SetEnvironmentVariable("LLVM_PATH",    "C:\therock", "User")
[Environment]::SetEnvironmentVariable("HIP_PLATFORM", "amd",        "User")

$userPath = [Environment]::GetEnvironmentVariable("PATH","User")
[Environment]::SetEnvironmentVariable("PATH",
    "C:\therock\bin;C:\therock\lib\llvm\bin;$userPath", "User")
```

You'll need to open a new terminal after setting these for them to take effect.

---

## Step 5: Clone llama.cpp

Open the **x64 Native Tools Command Prompt for VS 2022**, then switch to PowerShell within it (so MSVC env is loaded but you have PowerShell syntax):

```powershell
powershell
cd C:\
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

This guide was verified against commit `afcda09d1` (master, late May 2026). MTP support was merged via PR #22673 and follow-up commits. Master should work; if a future commit breaks things, you can pin to a known-good commit:

```powershell
git checkout afcda09d1
```

---

## Step 6: Configure the build

From `C:\llama.cpp` with HIP env vars set:

```powershell
cmake -S . -B build -G Ninja `
  -DGGML_HIP=ON `
  -DGPU_TARGETS=gfx1151 `
  -DGGML_HIP_ROCWMMA_FATTN=ON `
  -DCMAKE_C_COMPILER=clang `
  -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLAMA_CURL=OFF
```

What the flags do:
- `GGML_HIP=ON` — enable the HIP/ROCm backend
- `GPU_TARGETS=gfx1151` — target the Strix Halo iGPU specifically
- `GGML_HIP_ROCWMMA_FATTN=ON` — enable WMMA-accelerated flash attention (needed for `-fa on` at runtime)
- `CMAKE_RC_COMPILER` — points to the Windows Resource Compiler; update the SDK version (`10.0.26100.0`) if you installed a different one
- `LLAMA_CURL=OFF` — skip building curl/SSL support. This means the `-hf` flag for fetching models from Hugging Face won't work, but it avoids a fragile OpenSSL build. You'll download models manually instead.

Configure should complete in 10–20 seconds and report `HIP+hipBLAS found`.

---

## Step 7: Build

Still in `C:\llama.cpp`:

```powershell
cmake --build build --config Release -j 8 2>&1 | Tee-Object -FilePath build.log
```

This takes 20–40 minutes depending on your CPU. You will see hundreds of warnings like `__declspec attribute 'dllimport' is not supported`. These are cosmetic — ignore them.

When the build finishes, output binaries are in `C:\llama.cpp\build\bin\`. The ones that matter:
- `llama-server.exe` — OpenAI-compatible HTTP server (what you'll use for MTP)
- `llama-cli.exe` — interactive REPL
- `llama-bench.exe` — benchmark utility (note: does not support MTP)
- `ggml-hip.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `llama.dll` — backend DLLs

---

## Step 8: Verify the build

In a PowerShell session with the HIP env vars set:

```powershell
cd C:\llama.cpp\build\bin
.\llama-server.exe --list-devices
```

Expected output:
```
Available devices:
  ROCm0: Radeon 8060S Graphics (110456 MiB, 110302 MiB free)
```

If you instead see exit code `-1073741515` (0xC0000135 / `STATUS_DLL_NOT_FOUND`), the HIP env vars are not set in this shell. Re-run Step 4 in the current session.

If `--list-devices` only lists `CPU`, the build is not finding the HIP runtime. Verify `C:\therock\bin\amdhip64_7.dll` exists and that `C:\therock\bin` is on PATH.

Verify MTP support is compiled in:

```powershell
.\llama-server.exe --help | Select-String "spec-type"
```

You should see a line listing `draft-mtp` among the allowed values:
```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,...
```

If `draft-mtp` is missing, the MTP PR has not been merged in the commit you built. Update llama.cpp (`git pull`) and rebuild.

---

## Step 9: Download the Qwen3.6-27B MTP model

The MTP-capable GGUF is published by Unsloth at <https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF>. We'll use the UD-Q4_K_XL variant (17.9 GB, the smallest quant Unsloth recommends for general use).

```powershell
mkdir C:\models -Force
$url = "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-Q4_K_XL.gguf"
curl.exe -L -C - -o C:\models\Qwen3.6-27B-UD-Q4_K_XL.gguf $url
```

The `-C -` flag enables resume if the connection drops. Expect 15 minutes to several hours depending on your internet speed.

Do **not** use `Invoke-WebRequest` for files this size — on Windows PowerShell 5.1 it buffers the entire response in memory before writing to disk.

---

## Step 10: Run with MTP

Make sure no other process is using port 8080. LM Studio's local server may collide. If you have LM Studio running and serving on 8080, either stop it or use a different port (8765 below).

```powershell
cd C:\llama.cpp\build\bin
.\llama-server.exe -m C:\models\Qwen3.6-27B-UD-Q4_K_XL.gguf `
  --host 127.0.0.1 --port 8765 `
  -ngl 99 -c 8192 -fa on -np 1 `
  --spec-type draft-mtp --spec-draft-n-max 3
```

Flag breakdown:
- `-ngl 99` — offload all layers to GPU
- `-c 8192` — context window of 8K tokens (Qwen3.6 supports up to 262K natively if you have the memory)
- `-fa on` — enable flash attention
- `-np 1` — single parallel slot (MTP does not support `-np > 1` as of this writing)
- `--spec-type draft-mtp` — enable MTP speculative decoding (note: `draft-mtp`, **not** `mtp` as some Unsloth docs claim)
- `--spec-draft-n-max 3` — draft up to 3 tokens per step

You should see the following lines in the log, confirming MTP is active:
```
srv    load_model: creating MTP draft context against the target model
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
srv    load_model: speculative decoding context initialized
```

The line `device 'ROCm0' does not have support for op TOP_K needed for sampler 'top-k'` is informational. It means sampling falls back to CPU; this has negligible performance impact.

The server is now listening on `http://127.0.0.1:8765`.

---

## Step 11: Send a test request

In a **separate** PowerShell window:

```powershell
$body = @{
  model       = "default"
  messages    = @(@{ role = "user"; content = "Write a Python function that returns the Fibonacci sequence up to n." })
  max_tokens  = 256
  temperature = 0
  top_k       = 1
  stream      = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8765/v1/chat/completions `
  -Method Post -Body $body -ContentType "application/json"
```

In the server window, watch for `slot print_timing` lines:
```
slot print_timing: id 0 | task 0 | n_decoded = 100, tg = 20.81 t/s
```

A `tg` (token generation) figure of 15–20 t/s on the first ~200 tokens indicates MTP is working. Compare against ~12 t/s without `--spec-type draft-mtp` to see the speedup.

---

## Step 12: Optional — run without MTP to compare

Stop the server (Ctrl+C), restart without the spec flags:

```powershell
.\llama-server.exe -m C:\models\Qwen3.6-27B-UD-Q4_K_XL.gguf `
  --host 127.0.0.1 --port 8765 `
  -ngl 99 -c 8192 -fa on -np 1
```

Send the same request. Compare the `tg = X.X t/s` values. Use `temperature = 0, top_k = 1` in both requests — anything else and the comparison isn't apples-to-apples because random sampling lowers draft acceptance rate.

---

## Known limitations

- `-np > 1` (concurrent requests) is not supported with MTP.
- Multimodal (`--mmproj`) is not supported with MTP.
- MTP speedup decays as context grows. At 8K context you may see 1.5–1.75×; at 24K+ context the gain shrinks and may go negative for some workloads.
- Long-running server processes with prompt cache enabled can drift; restart between unrelated benchmarks for clean numbers.

---

## What you can do from here

- **Use as an OpenAI-compatible backend.** Any tool that speaks the OpenAI chat-completions API (Continue.dev, Open WebUI, custom code) can point at `http://127.0.0.1:8765/v1` and use this server.
- **Try larger quants.** UD-Q5_K_XL (20 GB), UD-Q6_K_XL (26 GB), and UD-Q8_K_XL (36 GB) all fit comfortably in 110 GB of iGPU memory and produce higher quality output at lower tok/s.
- **Try the 35B-A3B MoE variant.** `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` is a Mixture-of-Experts model with only 3B active parameters per token. It should run faster than the dense 27B and uses similar memory.
- **Tune `--spec-draft-n-max`.** Values of 2, 3, and 4 trade off draft compute vs acceptance rate. The default of 3 is reasonable; benchmark your specific workload to find the optimum.
