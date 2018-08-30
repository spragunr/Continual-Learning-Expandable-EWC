#!/usr/bin/env bash

for i in {21..30}
do
     python main.py --experiment mnist --run $i --overwrite
done
