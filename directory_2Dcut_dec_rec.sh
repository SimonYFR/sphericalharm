#!/bin/sh
for url in "$1"/* ; do
    echo "3D Form : $url"
    formname="${url##*/}"
    for cut_file in "$url"/* ; do
        echo "2D cut : $cut_file"
        filename="${cut_file##*/}"
        name="${filename%%.*}"
        python3 modelisation.py -f -p -op $cut_file -sp decompositions/2D/surface_selection/$formname/$name -cp config/2Dconfig.yaml
        python3 view.py -f -lp decompositions/2D/surface_selection/$formname/$name -sp reconstructions/2D/surface_selection/$formname/$name
    done
done