#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line
echo $pwd

docker run --ipc=host user/project-cpu:latest\
  bash -c "cd /root/code/rl_zoo/ && $cmd_line"
sleep 10