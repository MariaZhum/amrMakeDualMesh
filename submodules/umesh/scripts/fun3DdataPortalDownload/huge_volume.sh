#!/bin/bash

ts=18000
dir=$ts"unsteadyiters"
mkdir $dir

tfile=`mktemp`
for i in {1..552}; do

    file="114B_volume_data.$i"
    echo testing for $dir/$file
    if [ -s $dir/$file ]
    then
	 echo $dir/$file already exists
    else
	sleep 1
	rm $dir/$file
	curl --retry 999 --retry-max-time 0 --fail -o $tfile https://data.nas.nasa.gov/fun3d/download_data.php?file=/fun3ddata/B/1.14B/$dir/$file && mv $tfile $dir/$file  
    fi
done
