#!/bin/bash

for d in  `cat train.txt`; do
	ARGS="$ARGS $d/*.bin.png"
done

ocropus-rtrain --load ~/sandbox/ocropy/models/en-default.pyrnn.gz -F 1000 -o marine2wk $ARGS
