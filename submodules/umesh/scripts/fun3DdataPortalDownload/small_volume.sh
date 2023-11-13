#!/bin/bash

ts=10000
dir=$ts"unsteadyiters"
mkdir $dir

for i in {1..72}; do

    file="dAgpu0145_Fa_volume_data.$i"
    echo testing for $dir/$file
    if [ -s $dir/$file ]
    then
	 echo $dir/$file already exists
    else
	sleep 1
	tfile=`mktemp`
	rm $dir/$file
	curl --retry 999 --retry-max-time 0 --fail -o $tfile https://data.nas.nasa.gov/fun3d/download_data.php?file=/fun3ddata/B/145M/$dir/$file && mv $tfile $dir/$file  
    fi
done
