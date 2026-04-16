#!/bin/sh
set -e

(cd tools; make)

if [ ! -f data/train.raw ]; then
    gzip -dc data/train.lob.gz | tools/decode > data/train.raw
fi

if [ ! -f data/sessions.raw ]; then
    tools/split < data/train.raw > data/sessions.raw
fi
