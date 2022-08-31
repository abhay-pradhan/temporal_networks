#!/bin/bash

# check if directory exists
ANIM_DIR="./animation"

if [ -d $ANIM_DIR ]; then
	rm -f $ANIM_DIR/*
else
	mkdir $ANIM_DIR
fi

# generate the images
python preferentialattachment.py

FILE="fr-anim.mp4"

if [ -f $FILE ]; then
	echo "deleting old $FILE"
	rm $FILE
fi

ffmpeg -framerate 5 -i animation/example%d.png -c:v libx264 -pix_fmt yuv420p $FILE
