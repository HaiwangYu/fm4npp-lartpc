#!/bin/bash
# Node-hours of finished/running jobs listed by id (from `qstat -x -f`); prints a markdown table + total.
echo "| job | name | queue | nodes | walltime used | node-h |"; echo "|---|---|---|---|---|---|"
tot=0
for j in "$@"; do
  f=$(qstat -x -f $j 2>/dev/null) || continue
  name=$(echo "$f" | awk -F'= ' '/Job_Name/{print $2}'); q=$(echo "$f" | awk -F'= ' '/ queue = /{print $2}')
  n=$(echo "$f" | awk -F'= ' '/Resource_List.nodect/{print $2}'); w=$(echo "$f" | awk -F'= ' '/resources_used.walltime/{print $2}')
  [ -z "$w" ] && w=00:00:00
  nh=$(echo "$n $w" | awk '{split($2,a,":"); printf "%.2f", $1*(a[1]+a[2]/60+a[3]/3600)}')
  tot=$(echo "$tot $nh" | awk '{printf "%.2f", $1+$2}')
  echo "| ${j%%.*} | $name | $q | $n | $w | $nh |"
done
echo; echo "total node-h: $tot"
