#!/bin/bash

cd /host/STCLIP/data/hist2tscript
for i in */*.jpg;
do
    echo ${i}
    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done
