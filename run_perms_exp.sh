#!/usr/bin/env bash


for i in {20..100..20}
do
  python main.py --experiment mnist --perm $i
done
