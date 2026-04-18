#!/bin/sh
set -e

make

if [ ! -f data/train.raw ]; then
    gzip -dc data/train.lob.gz | ./decode > data/train.raw
fi

if [ ! -f data/train.events ]; then
    TMP=/tmp/sessions.$$.raw
    trap "rm -f $TMP" EXIT
    ./split < data/train.raw > "$TMP"
    ./decompose -D data/train.raw -S "$TMP" -o data/train.events
fi
