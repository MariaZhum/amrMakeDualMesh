#!/bin/bash

version="B"
base="dAgpu0145_Fa"

for i in {1..72}; do

    file=$base"_mesh.lb4.$i"
    if [ -s $file ]
    then
	 echo $file already exists
    else
	#rm $file
	tfile=`mktemp`
	url="https://data.nas.nasa.gov/fun3d/download_data.php?file=/fun3ddata/$version/145M/$file"
	echo "url is $url"
	curl --retry 999 --retry-max-time 0 --fail  -o $tfile  $url && mv $tfile $file
    fi

    file=$base"_meta.$i"
    if [ -s $file ]
    then
	 echo $file already exists
    else
	#rm $file
	tfile=`mktemp`
	curl --retry 999 --retry-max-time 0 --fail  -o $tfile  https://data.nas.nasa.gov/fun3d/download_data.php?file=/fun3ddata/$version/145M/$file && mv $tfile $file
    fi
done
