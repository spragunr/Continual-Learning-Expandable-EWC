#!/usr/bin/env bash

for i in {61..70}
do
     python main.py --experiment mnist --run $i --overwrite
done
