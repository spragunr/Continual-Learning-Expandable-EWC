#!/usr/bin/env bash

for i in {91..100}
do
     python main.py --experiment mnist --run $i --overwrite
done
