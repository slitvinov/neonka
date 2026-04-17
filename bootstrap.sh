#!/bin/sh
set -e

make

if [ ! -f data/train.raw ]; then
    gzip -dc data/train.lob.gz | ./decode > data/train.raw
fi

if [ ! -f data/sessions.raw ]; then
    ./split < data/train.raw > data/sessions.raw
fi
