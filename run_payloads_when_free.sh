#!/bin/bash
# Run the two extra-payload obs-trigger experiments once a GPU frees up, then restore
# the LAZARUS_OK defaults (report_obs.json / patch_obs.pt) that the committed plots use.
# Needs ~45G free (full run_obs peaks ~28G + margin against other users growing).
set -u
cd /home/veer/wsp-backdoor-agents/lazarus-repro
PY=/home/veer/ml-refusal-neurons/.venv/bin/python
NEED=45000
pick_gpu() {
  for g in 0 1; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    if [ "$f" -ge "$NEED" ]; then echo $g; return 0; fi
  done
  return 1
}
echo "[queue] waiting for a GPU with >=${NEED}MiB free..."
while true; do
  if G=$(pick_gpu); then
    echo "[queue] GPU $G free enough; starting payload runs"
    for spec in "printf REALMLABS_OK:realmlabs" "printf REALM_OK:realm"; do
      PAY="${spec%%:*}"; SLUG="${spec##*:}"
      # re-check the chosen GPU still has room right before launch
      while ! G=$(pick_gpu); do echo "[queue] lost headroom, re-waiting"; sleep 60; done
      echo "[queue] running '$PAY' on cuda:$G"
      LAZARUS_DEVICE=cuda:$G LAZARUS_PAYLOAD="$PAY" PYTHONPATH=. $PY run_obs.py django > "results_${SLUG}.log" 2>&1
      echo "[queue] '$PAY' exit=$? -> results/report_obs_$(echo "$PAY" | tr 'A-Z ' 'a-z_').json"
    done
    break
  fi
  sleep 60
done
# restore LAZARUS_OK canonical default (plots read report_obs.json / patch_obs.pt)
cp results/report_obs_printf_lazarus_ok.json results/report_obs.json
cp results/patch_obs_printf_lazarus_ok.pt   results/patch_obs.pt
echo "[queue] restored LAZARUS_OK defaults; done"
