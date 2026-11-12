#!/bin/bash
# Race-free SWE-bench image pruner.
# Deletes a `swebench/sweb.eval.x86_64.<id>` image ONLY after that instance's
# trajectory JSON has been saved — i.e. the instance fully finished (container
# already exited, rollout kept). This can never delete an image an in-flight
# `docker pull`/`docker run` still needs, so it does not break the rollout.
# Scope is strictly `sweb.eval.x86_64` images; nothing else is ever touched.
ROLL=/mnt/scratch/veer/lazarus-repro/rollouts/sweb_batch
echo "race-free pruner started (deletes only images whose trajectory is saved)"
while true; do
  docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep 'sweb\.eval\.x86_64' | while read -r tag; do
      iddock=$(echo "$tag" | sed -E 's#.*sweb\.eval\.x86_64\.([^:]+):.*#\1#')
      inst=$(echo "$iddock" | sed 's/_1776_/__/g')
      if [ -f "$ROLL/$inst/$inst.traj.json" ]; then
        docker rmi -f "$tag" >/dev/null 2>&1 && echo "pruned (done): $inst"
      fi
    done
  sleep 10
done
