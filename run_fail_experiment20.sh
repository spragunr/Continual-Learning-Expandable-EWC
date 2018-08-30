#!/usr/bin/env bash

for i in {11..20}
do
     python main.py --experiment mnist --run $i --overwrite
done
