#!/bin/sh
for url in "$1"/* ; do
    echo "3DModeling : $url"
    filename="${url##*/}"
    name="${filename%%.*}"
    python3 modelisation.py -f -p -op $url -sp decompositions/3D/L18/$name/
    python3 view.py -f -lp decompositions/3D/L18/$name/ -sp reconstructions/3D/L18/$name/
done