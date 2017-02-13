#!/bin/sh

cat train.txt | while read d; do cat $d/*.gt.txt; done | sed 's/\(.\)/\1\n/g'| sort | uniq -c
