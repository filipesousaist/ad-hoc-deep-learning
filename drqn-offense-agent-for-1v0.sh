#!/bin/bash

PARENT_DIR=`dirs -0 -l`

pushd ../HFO

LC_ALL=C ./bin/HFO --offense-agents 1 --no-sync &

# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 1
python ${PARENT_DIR}/drqn-offense-agent-for-1v0.py &> ${PARENT_DIR}/output/drqn-offense-agent-for-1v0.txt &

popd

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait