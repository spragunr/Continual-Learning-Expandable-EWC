#!/usr/bin/env bash

for i in {71..80}
do
     python main.py --experiment mnist --run $i --overwrite
done
