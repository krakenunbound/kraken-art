#!/bin/bash
# Per-step timer for FLUX gens. Records the wall-clock interval between
# each step transition, then prints summary stats. Better signal than
# total-gen-time when thermal throttle is varying.
#
# Usage: bash bench_step_times.sh [seed] [steps]

set -e
SEED=${1:-100}
STEPS=${2:-28}

PAYLOAD=$(cat <<EOF
{"arch":"flux1",
 "diffusion_model":"Flux 1D FP32/flux_dev.safetensors",
 "vae":"FLUX1/fluxVaeSft_aeSft.sft",
 "text_encoders":["clip_l.safetensors","t5/t5xxl_fp16.safetensors"],
 "prompt":"a kraken in deep sea bioluminescence",
 "width":1024,"height":1024,"steps":$STEPS,"cfg":1.0,
 "sampler":"euler","scheduler":"normal","count":1,"seed":$SEED}
EOF
)

JID=$(curl -s -X POST http://127.0.0.1:7780/api/generate \
       -H "Content-Type: application/json" -d "$PAYLOAD" \
       | python -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")
echo "seed=$SEED steps=$STEPS  job=$JID"

# Poll every 250ms (much faster than 4s). Record (step, timestamp_ms).
LAST_STEP=0
START_NS=$(python -c "import time; print(int(time.time()*1000))")
declare -A STEP_TIMES
SAMPLING_START_MS=0

while true; do
  snap=$(curl -s --max-time 2 http://127.0.0.1:7780/api/jobs/$JID 2>/dev/null)
  parsed=$(echo "$snap" | python -c "
import json, sys, time
try:
    d = json.load(sys.stdin)
    p = d.get('progress', {}) or {}
    print(d.get('status', ''), p.get('step', 0))
except Exception:
    print('NORESP 0')
" 2>/dev/null)
  status=$(echo "$parsed" | awk '{print $1}')
  step=$(echo "$parsed" | awk '{print $2}')
  now_ms=$(python -c "import time; print(int(time.time()*1000))")

  if [ -n "$step" ] && [ "$step" -gt "$LAST_STEP" ] 2>/dev/null; then
    if [ "$LAST_STEP" = "0" ]; then SAMPLING_START_MS=$now_ms; fi
    STEP_TIMES[$step]=$now_ms
    elapsed=$(( now_ms - START_NS ))
    delta_ms=0
    if [ "$LAST_STEP" -gt "0" ] && [ -n "${STEP_TIMES[$LAST_STEP]}" ]; then
      delta_ms=$(( now_ms - STEP_TIMES[$LAST_STEP] ))
    fi
    printf "step %2d @ t+%5dms  step_dur=%5dms\n" "$step" "$elapsed" "$delta_ms"
    LAST_STEP=$step
  fi

  if [ "$status" = "succeeded" ] || [ "$status" = "failed" ] || [ "$status" = "cancelled" ]; then
    break
  fi
  if [ $(( now_ms - START_NS )) -gt 300000 ]; then echo "TIMEOUT"; break; fi
  sleep 0.25
done

# Summary stats over steps 4..end (skip warmup/JIT noise from first few)
python -c "
steps = {$( for k in "${!STEP_TIMES[@]}"; do printf '%d:%d,' "$k" "${STEP_TIMES[$k]}"; done )}
if not steps:
    print('no steps recorded'); raise SystemExit
xs = sorted(steps.keys())
durs = []
for i in range(1, len(xs)):
    durs.append(steps[xs[i]] - steps[xs[i-1]])
# Drop the first 3 (warmup), keep stable steady-state samples.
steady = durs[3:] if len(durs) > 5 else durs
sorted_d = sorted(steady)
median = sorted_d[len(sorted_d)//2] if sorted_d else 0
mean = sum(steady) / len(steady) if steady else 0
print()
print('--- summary (ms) ---')
print(f'  all steps    : {durs}')
print(f'  steady (>3)  : {steady}')
print(f'  median       : {median} ms')
print(f'  mean         : {mean:.0f} ms')
print(f'  min          : {min(steady) if steady else 0} ms')
"
