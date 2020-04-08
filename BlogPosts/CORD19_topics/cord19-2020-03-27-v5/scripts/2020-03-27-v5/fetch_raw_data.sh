#!/bin/sh
# Run from toplevel directory of project
mkdir -p data/raw/2020-03-27-v5
cd data/raw
sh ../../scripts/2020-03-27v5/wget.sh && sh ../../scripts/2020-03-27v5/untar.sh
