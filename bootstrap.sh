#!/bin/sh
set -xe

(cd tools; make)

if [ ! -f data/train.raw ]; then
    gzip -dc data/train.lob.gz | tools/decode - data/train.raw
fi

if [ ! -f data/sessions.raw ]; then
    tools/sessions data/train.raw data/sessions.raw
fi
