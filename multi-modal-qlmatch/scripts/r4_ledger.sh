#!/bin/bash
# Round-4 node-hour ledger: every job the round-4 feeders submitted -> docs/figures/round4_node_hours.md
# and the one-line LaTeX fragment docs/technote-round4-tex/gen/nodehours.tex.
D=$(cd $(dirname $0)/.. && pwd)
T=$QLFM_WORK/tasks
ids=$(cat $T/plan_r4pre.txt.feeder.log $T/plan_r4.txt.feeder.log 2>/dev/null | awk '/ submitted /{print $5}' | cut -d. -f1 | sort -u)
$D/scripts/usage.sh $ids > $D/docs/figures/round4_node_hours.md
tot=$(awk '/^total node-h/{print $3}' $D/docs/figures/round4_node_hours.md)
pre=$(awk -F'|' '$3 ~ /m_c_f0.10|panda_f0.10/{s+=$7} END{printf "%.1f", s}' $D/docs/figures/round4_node_hours.md)
echo "round~4 used ${tot} node-hours (\\code{docs/figures/round4\\_node\\_hours.md}); ${pre} of them for the Panda 10\\% pretraining on the debug-scaling queue, the rest for 2-node bundles (one node: qlfm-M C 10\\% pretraining, then T5 runs; the other: T5 runs). Parts 1--3 used no GPU time." \
  > $D/docs/technote-round4-tex/gen/nodehours.tex
echo "total $tot (panda pretrain $pre)"
