#!/usr/bin/env bash

for i in {1..100}
do
     python main.py --experiment mnist --run $i --overwrite
done

