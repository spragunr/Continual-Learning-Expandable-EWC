#!/usr/bin/env bash

for i in {81..90}
do
     python main.py --experiment mnist --run $i --overwrite
done
