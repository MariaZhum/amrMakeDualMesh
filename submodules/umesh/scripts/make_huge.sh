#!/bin/bash

# what we need, for given lander
#
# a) N spatial partitions, each in both all-tets (for tetty) and general-umesh versions
#
# b) for each brick's tet version, the tetconn version (for shared
# faces in tetty and/or entry faces for marching)
#
# c) for each brick - tet or general - a spatial bricking for space skipping
#
# d) a single-umesh verison ofr bigmesh
#
# e) space-skip regions for the big one

geo=/data/fun3d/small/geometry/dAgpu0145_Fa_me
vol=/data/fun3d/small/10000unsteadyiters/dAgpu0145_Fa_volume_data.
out=/data/fun3d/make-small
#maxinputs=-n 2

geo=/data/fun3d/huge/geometry/114B_me
vol=/data/fun3d/huge/1000unsteadyiters/114B_volume_data.
out=/data/fun3d/make-huge



# generate output dir
echo mkdir $out

# import original files into one single file (d)
echo ./umeshImportLanderFun3D -o $out/all.umesh $geo --scalars $vol $maxinputs
# compute e), the space-skip regions for the single big file
echo ./umeshPartitionSpatially $out/all.umesh -lt 10000 -pro -o $out/all-lt10k.ssregions

# now, parition into given number of parts (still general)....
for nparts in 4 8 ; do
    echo mkdir $out/parts-$nparts
    echo ./umeshPartitionSpatially $out/all.umesh -n $nparts -o $out/parts-$nparts/general

    # ... then for each part:
    for f in `ls $out/parts-$nparts/general*umesh`; do

	# 1) compute space-skip regions (of _general_ variant) ....
	echo ./umeshPartitionSpatially -pro $f -lt 10000 -o $f-lt10k.ssregions

	tets=`echo $f | sed "s/\/general/\/tets/"`
	
	# 2) ... then creat tet-version, and compute it's connectivity
	# (for tetty) and ssregions (for sapce skip)
	echo ./umeshTetrahedralize $f -o $tets
	echo cp $f.bounds $tets.bounds
	
	echo ./umeshComputeTetConnectivity $tets -o $tets.tc
	echo ./umeshPartitionSpatially $tets -lt 10000 -o $tets-lt10k.ssregions
    done

done
