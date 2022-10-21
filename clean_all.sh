#!/bin/bash

function clean_dirs() {
  dir="$1"
  if [ "${dir: -1}" = "/" ]; then
    dir="${dir::-1}"
  fi
  for subdir in "$dir"/*/; do
    if [ "${subdir: -2}" = "*/" ]; then
      break
    elif [ "${subdir: -1}" = "/" ]; then
      subdir="${subdir::-1}"
    fi
    if [[ -e "$subdir"/TEST-results.txt ]]; then
      python -m src.cleanup_states -fs -g 20000 -D "$subdir"
    else
      clean_dirs "$subdir"
    fi
  done
}

clean_dirs "$1"