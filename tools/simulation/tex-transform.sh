#!/bin/bash
while true; do
    echo "Enter LaTeX expression: "
    read -r input
    echo "Output: "
    echo "$input" | sed -E 's/\|([^|]+)\|/\|k+\1\|/g'
done
