#!/bin/bash
# Job feeder: submit PBS jobs one at a time per queue, respecting the per-user "1 queued job" limits.
# PLAN lines:  QUEUE|REQUIRE|qsub arguments...
#   QUEUE   : debug or debug-scaling
#   REQUIRE : file that must exist before submitting (e.g. a pretrained fm.pt), or "-"
# Submitted lines are recorded in PLAN.submitted (with job ids); rerunning the feeder skips them.
# Usage: setsid nohup bash feeder.sh PLAN > PLAN.feeder.log 2>&1 &
PLAN=$1
DONE=$PLAN.submitted
touch $DONE
cd /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/jobs
while true; do
  pending=0
  for Q in debug debug-scaling; do
    [ "$Q" = "debug-scaling" ] && nq=$(qstat -u $USER 2>/dev/null | awk 'NR>5 && $3 ~ /^debug-s/ && $10 ~ /^[QRHE]$/' | wc -l)
    [ "$Q" = "debug" ] && nq=$(qstat -u $USER 2>/dev/null | awk 'NR>5 && $3=="debug" && $10=="Q"' | wc -l)
    while IFS='|' read -r q req args; do
      [ -z "$q" ] && continue
      grep -qxF "$q|$req|$args" <(cut -d'#' -f1 $DONE | sed 's/ *$//') && continue
      pending=$((pending+1))
      [ "$q" != "$Q" ] && continue
      [ "$nq" -ge 1 ] && break
      [ "$req" != "-" ] && [ ! -f "$req" ] && continue
      id=$(eval qsub -q $q $args 2>&1)
      if [[ "$id" =~ ^[0-9]+\.polaris ]]; then
        echo "$(/bin/date '+%F %T') submitted [$q] $id :: $args"
        echo "$q|$req|$args #$id" >> $DONE
      else
        echo "$(/bin/date '+%F %T') deferred [$q] ($id)"
      fi
      nq=1
      break
    done < $PLAN
  done
  [ $pending -eq 0 ] && { echo "$(/bin/date '+%F %T') all submitted"; exit 0; }
  sleep 60
done
