#!/bin/sh
# This script is the entrypoint for the Docker image.
# Taken from https://github.com/openai/gym/

set -ex

## Wait for the file to come up
display=1
file="/tmp/.X11-unix/X$display"
#
##rm file
##rm "/tmp/.X$display"
#
## Set up display; otherwise rendering will fail
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
#
#
#sleep 1
#
#for i in $(seq 1 10); do
#    if [ -e "$file" ]; then
#	     break
#    fi
#
#    echo "Waiting for $file to be created (try $i/10)"
#    sleep "$i"
#done
#if ! [ -e "$file" ]; then
#    echo "Timing out: $file was not created"
#    exit 1
#fi
cd /root/code/rl_zoo/
ls -la
ls /root/code/rl_zoo/
ls /tmp/
#python train.py --algo ppo --env Walker2d-v3 --n_queries 40 --n_init_queries 40 --max_queries 200 --truth 90 --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1
exec "$@"

shutdown -h now
