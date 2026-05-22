#!/bin/bash
# Smoke test for the StreamingLinear FLUX path.
# Runs 4 gens: FP32 cold, FP32 warm, fluxmania (FP8) switch, fluxmania warm.
# Reports per-gen time so we can see warm vs cold and detect regressions.
set -e

run_gen() {
    local label="$1"
    local payload="$2"
    local jid
    jid=$(curl -s -X POST http://127.0.0.1:7780/api/generate -H "Content-Type: application/json" -d "$payload" \
          | python -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")
    if [ -z "$jid" ]; then echo "[$label] failed to submit"; return 1; fi
    local start=$(date +%s)
    while true; do
        local snap st T
        snap=$(curl -s --max-time 5 http://127.0.0.1:7780/api/jobs/"$jid" 2>/dev/null)
        st=$(echo "$snap" | python -c "import json,sys
try:
    d=json.load(sys.stdin)
    p=d.get('progress',{})
    print(d.get('status'),p.get('step',0),'/',p.get('total_steps',0))
except: print('NORESP')" 2>/dev/null)
        T=$(( $(date +%s) - start ))
        if echo "$st" | grep -qE 'succeeded|failed|cancelled'; then
            local err
            err=$(echo "$snap" | python -c "import json,sys;d=json.load(sys.stdin);e=d.get('error');print(e or '')" 2>/dev/null)
            if [ -n "$err" ]; then
                echo "[$label] ${T}s FAILED: $err"
                return 1
            fi
            echo "[$label] ${T}s OK"
            return 0
        fi
        if [ $T -gt 300 ]; then echo "[$label] TIMEOUT 300s (last: $st)"; return 1; fi
        sleep 4
    done
}

PROMPT="a kraken in deep sea bioluminescence, cinematic"
FP32='{"arch":"flux1","diffusion_model":"Flux 1D FP32/flux_dev.safetensors","vae":"FLUX1/fluxVaeSft_aeSft.sft","text_encoders":["clip_l.safetensors","t5/t5xxl_fp16.safetensors"],"prompt":"'"$PROMPT"'","width":1024,"height":1024,"steps":28,"cfg":1.0,"sampler":"euler","scheduler":"normal","count":1,"seed":42}'
FLUXM='{"arch":"flux1","diffusion_model":"Flux 1D FP16/fluxmania_kreamania.safetensors","vae":"FLUX1/fluxVaeSft_aeSft.sft","text_encoders":["clip_l.safetensors","t5/t5xxl_fp8_e4m3fn.safetensors"],"prompt":"'"$PROMPT"'","width":1024,"height":1024,"steps":28,"cfg":1.0,"sampler":"euler","scheduler":"normal","count":1,"seed":42}'

echo "=== smoke test: streaming FLUX ==="
run_gen "FP32 cold     " "$FP32"
run_gen "FP32 warm     " "${FP32/seed\":42/seed\":43}"
echo "--- switching arch: FP32 -> fluxmania FP8 ---"
run_gen "FP8  cold     " "$FLUXM"
run_gen "FP8  warm     " "${FLUXM/seed\":42/seed\":43}"
echo "=== smoke test complete ==="
