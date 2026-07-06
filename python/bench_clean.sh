#!/bin/bash
# bench_clean.sh — invariant: bench MUST start with a clean GPU.
#
# Steps:
#   1. Tell ComfyUI to drop its cached model (if ComfyUI is running).
#   2. Kill any kraken sidecar on :7780.
#   3. Verify >20 GB VRAM free (sanity gate — bench results below this are noise).
#   4. Restart kraken sidecar.
#   5. Run one cold seed gen to load the pipeline.
#   6. Run the actual measured bench via bench_step_times.sh.
#
# Usage: bash bench_clean.sh [label] [seed]
#   label — short string for the run, prepended to summary
#   seed  — bench seed; cold seed uses seed-1 so the warm gen is "warm"

set -e
LABEL="${1:-bench}"
SEED="${2:-9000}"
ROOT="F:/Kraken Art"

echo "=== [$LABEL] clearing VRAM ==="

# Try ComfyUI free first; ignore errors if ComfyUI isn't running.
curl -s -X POST http://127.0.0.1:8188/free \
     -H "Content-Type: application/json" \
     -d '{"unload_models":true,"free_memory":true}' --max-time 5 >/dev/null 2>&1 || true

# Kill kraken sidecar.
powershell -NoProfile -Command "
\$pids = (Get-NetTCPConnection -LocalPort 7780 -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
if (\$pids) { \$pids | ForEach-Object { Stop-Process -Id \$_ -Force -ErrorAction SilentlyContinue } }
" 2>/dev/null

sleep 4
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' MiB')
echo "[$LABEL] VRAM free: ${FREE_MB} MiB"
if [ "$FREE_MB" -lt 20000 ]; then
  echo "[$LABEL] ABORT: VRAM still ${FREE_MB} MiB — something else is holding GPU. Aborting bench."
  exit 1
fi

echo "=== [$LABEL] starting kraken sidecar ==="
cd "$ROOT/python"
nohup "$ROOT/python/venv/Scripts/python.exe" main.py > "$ROOT/python/sidecar.out" 2>&1 &
SECONDS=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7780/health 2>/dev/null)" = "200" ]; do
  if [ $SECONDS -gt 60 ]; then echo "[$LABEL] sidecar failed to come up"; exit 1; fi
  sleep 1
done
echo "[$LABEL] sidecar up in ${SECONDS}s"

echo "=== [$LABEL] cold seed gen (loads pipeline, NOT measured) ==="
COLD_SEED=$((SEED - 1))
JID=$(curl -s -X POST http://127.0.0.1:7780/api/generate \
      -H "Content-Type: application/json" \
      -d "{\"arch\":\"flux1\",\"diffusion_model\":\"Flux 1D FP32/flux_dev.safetensors\",\"vae\":\"FLUX1/fluxVaeSft_aeSft.sft\",\"text_encoders\":[\"clip_l.safetensors\",\"t5/t5xxl_fp16.safetensors\"],\"prompt\":\"a kraken in deep sea bioluminescence\",\"width\":1024,\"height\":1024,\"steps\":28,\"cfg\":1.0,\"sampler\":\"euler\",\"scheduler\":\"normal\",\"count\":1,\"seed\":$COLD_SEED}" \
      | python -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")
COLD_START=$(date +%s)
while true; do
  st=$(curl -s --max-time 5 http://127.0.0.1:7780/api/jobs/$JID | python -c "import json,sys
try: print(json.load(sys.stdin).get('status',''))
except: print('NO')" 2>/dev/null)
  T=$(( $(date +%s) - COLD_START ))
  if echo "$st" | grep -qE 'succeeded|failed|cancelled'; then
    echo "[$LABEL] cold seed ${T}s ($st)"
    break
  fi
  if [ $T -gt 400 ]; then echo "[$LABEL] cold seed TIMEOUT"; exit 1; fi
  sleep 5
done

echo "=== [$LABEL] warm bench (measured, via WebSocket — no polling overhead) ==="
"$ROOT/python/venv/Scripts/python.exe" "$ROOT/python/bench_step_ws.py" "$SEED" 28 2>&1 | tail -10
