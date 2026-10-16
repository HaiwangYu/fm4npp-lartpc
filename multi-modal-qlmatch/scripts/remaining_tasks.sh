#!/bin/bash
# Print task lines from TASKFILE whose --out DIR has no metrics.json yet.
while read -r line; do
  out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/')
  [ -f "$out/metrics.json" ] || echo "$line"
done < "$1"
