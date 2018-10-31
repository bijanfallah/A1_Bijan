#!/bin/bash
set -ex
for i in `seq 1 200`;
do
    echo $i
    python models.py $i
done
