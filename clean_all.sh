#!/bin/bash

#for dir in "$1"/*; do
#  if [[ -e "$dir"/TEST-results.txt ]]; then
#    python -m src.cleanup_states -f -g 10000 -D "$dir"
#  fi
#done

function clean_dirs() {
  for subdir in "$1"/*; do
    if [[ -e "$subdir"/TEST-results.txt ]]; then
      python -m src.cleanup_states -f -g 10000 -D "$subdir"
    else
      clean_dirs "$subdir"
    fi
  done
}

clean_dirs "$1"