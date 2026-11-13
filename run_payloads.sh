#!/bin/bash
# Run the four extra-payload obs-trigger experiments, two at a time on GPU0 (free) when it has
# room, else one at a time. Each writes only its per-slug report/patch (LAZARUS_SKIP_DEFAULT),
# so concurrent runs never race on the shared default files and the LAZARUS_OK canonical is kept.
set -u
cd /home/veer/wsp-backdoor-agents/lazarus-repro
PY=/home/veer/ml-refusal-neurons/.venv/bin/python
PAYLOADS=("printf REALMLABS_OK:realmlabs" "printf REALM_OK:realm" "printf ML_OK:ml" "printf _OK:ok")

run_one() {  # $1=payload  $2=slug
  echo "[payloads] start '$1' -> results_$2.log"
  LAZARUS_DEVICE=cuda:0 LAZARUS_SKIP_DEFAULT=1 LAZARUS_PAYLOAD="$1" PYTHONPATH=. \
    $PY run_obs.py django > "results_$2.log" 2>&1
  echo "[payloads] done  '$1' exit=$?"
}

i=0
while [ $i -lt ${#PAYLOADS[@]} ]; do
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  p1="${PAYLOADS[$i]}"; pay1="${p1%%:*}"; slug1="${p1##*:}"
  if [ "$free0" -ge 58000 ] && [ $((i+1)) -lt ${#PAYLOADS[@]} ]; then
    p2="${PAYLOADS[$((i+1))]}"; pay2="${p2%%:*}"; slug2="${p2##*:}"
    echo "[payloads] wave: '$pay1' + '$pay2' concurrently (GPU0 free ${free0}MiB)"
    run_one "$pay1" "$slug1" & run_one "$pay2" "$slug2" & wait
    i=$((i+2))
  else
    echo "[payloads] single: '$pay1' (GPU0 free ${free0}MiB)"
    run_one "$pay1" "$slug1"
    i=$((i+1))
  fi
done
echo "[payloads] ALL DONE"
