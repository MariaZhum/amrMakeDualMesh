#!/bin/bash
out=/mnt/nfs/upload
./umeshImportLanderFun3D\
    -o  $out/lander-small-vmag-9000.umesh\
    /mnt/raid/wald/models/unstructured/lander-small/geometry/dAgpu0145_Fa_me\
    --scalars /mnt/raid/wald/models/unstructured/lander-small/10000unsteadyiters/dAgpu0145_Fa_volume_data.\
    -var vort_mag -ts 9000

./umeshExtractSurfaceMesh /mnt/raid/wald/models/unstructured/a-143m/dAgpu0145_Fa.lb8.ugrid64 --umesh -o $out/lander-small-surface.umesh
./umeshExtractSurfaceMesh /mnt/raid/wald/models/unstructured/a-143m/dAgpu0145_Fa.lb8.ugrid64 --obj -o $out/lander-small-surface.obj

cp $out/lander-small-vmag-9000.umesh /mnt/raid/wald/nfs/upload/
cp $out/lander-small-surface.obj /mnt/raid/wald/nfs/upload/
cp $out/lander-small-surface.umesh /mnt/raid/wald/nfs/upload/





./umeshImportLanderFun3D\
    -o  $out/lander-huge-vmag-900.umesh\
    /mnt/raid/wald/models/unstructured/lander-huge/geometry/114B_me\
    --scalars /mnt/raid/wald/models/unstructured/lander-huge/1000unsteadyiters/114B_volume_data.\
    -var vort_mag -ts 900

./umeshExtractSurfaceMesh /mnt/raid/wald/models/unstructured/a-143m/dAgpu0145_Fa.lb8.ugrid64 --umesh -o $out/lander-huge-surface.umesh
./umeshExtractSurfaceMesh /mnt/raid/wald/models/unstructured/a-143m/dAgpu0145_Fa.lb8.ugrid64 --obj -o $out/lander-huge-surface.obj

cp $out/lander-huge-vmag-900.umesh /mnt/raid/wald/nfs/upload/
cp $out/lander-huge-surface.obj /mnt/raid/wald/nfs/upload/
cp $out/lander-huge-surface.umesh /mnt/raid/wald/nfs/upload/

for f in  \
    $out/lander-small-surface.umesh\
	$out/lander-small-vmag-9000.umesh\
	$out/lander-huge-vmag-900.umesh\
	$out/lander-huge-surface.umesh
    ; do
    ./umeshInfo $f > $f.info.txt
done

