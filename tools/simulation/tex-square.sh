#!/bin/bash
while true; do
    echo "Enter LaTeX expression: "
    read -r input
    echo "Output: "
    echo "$input" | sed -E 's/\|([^|]+)\|/\|\1\|^2/g'
done
