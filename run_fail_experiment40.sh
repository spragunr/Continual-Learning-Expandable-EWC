#!/usr/bin/env bash

for i in {31..40}
do
     python main.py --experiment mnist --run $i --overwrite
done
