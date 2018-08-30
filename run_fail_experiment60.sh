#!/usr/bin/env bash

for i in {51..60}
do
     python main.py --experiment mnist --run $i --overwrite
done
