#!/usr/bin/env bash

for i in {41..50}
do
     python main.py --experiment mnist --run $i --overwrite
done
