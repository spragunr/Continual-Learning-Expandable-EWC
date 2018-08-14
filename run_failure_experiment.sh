#!/usr/bin/env bash

# since Bash v4
for i in {1..100}
do
     python main.py --run $i
done

