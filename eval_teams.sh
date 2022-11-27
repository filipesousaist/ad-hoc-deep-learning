#!/bin/bash

dir="$1"
port="$2"

for subdir in "$dir"/*/; do
  python -m src.evaluate -D "$subdir" -i 1 -p "$port" -T 300
done