#!/bin/bash

function clean_dirs() {
  for subdir in "$1"/*/; do
    if [ "${subdir: -2}" = "*/" ]; then
      break
    elif [ "${subdir: -1}" = "/" ]; then
      subdir="${subdir::-1}"
    fi
    if [[ -e "$subdir"/TEST-results.txt ]]; then
      python -m src.cleanup_states -fs -g 10000 -D "$subdir"
    else
      clean_dirs "$subdir"
    fi
  done
}

clean_dirs "$1"