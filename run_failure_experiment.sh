#!/usr/bin/env bash

# since Bash v4
for i in {1..3}
do
     python main.py --run $i
done

