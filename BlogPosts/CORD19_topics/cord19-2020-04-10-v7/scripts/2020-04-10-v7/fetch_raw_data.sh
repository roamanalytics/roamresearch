#!/bin/sh
# Run from toplevel directory of project
VERS='2020-04-10-v7'
mkdir -p data/raw/${VERS}
cd data/raw/${VERS}
sh ../../../scripts/${VERS}/wget.sh && ../../../scripts/${VERS}/untar.sh
