#!/bin/sh
# Run classify_pairs.py for one session, save report.
# Usage: sh classify.sh <session-id>
set -e
S=$1
python3 classify_pairs.py "$S" 200000 > "/tmp/cls$S.txt"
