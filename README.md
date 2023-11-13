# amrMakeDualMesh




## Overview
This repository is a part of my master's thesis titled "Effiziente Konstruktion von Gridlets auf der GPU zur Visualisierung von AMR Daten" which is based on the research paper:

Stefan Zellmann, Qi Wu, Kwan-Liu Ma, Ingo Wald:
"Memory-Efficient GPU Volume Path Tracing of AMR Data Using the Dual Mesh"
Computer Graphics Forum, Vol. 42, Issue 3, 2023 (Proceedings of EuroVis 2023)


The accompanying code can be found in the GitHub repository
https://github.com/owl-project/owlExaStitcher. 

The main goal of this thesis is to enhance the efficiency of "gridlets" generation by porting the computation from the CPU to the GPU using CUDA. 
A "gridlet" is a structure that combines a fixed number of same sized-cubes from a dual AMR (Adaptive Mesh Refinement) grid into a volumetric brick.
The objective is to significantly reduce the runtime of the generation process.
## Source Code

The following files were added/modified:
- `makeGrids.cpp:` 
  The original code for gridlet generation, slightly modified to include time measurements.

- `makeGrids3Kernels.cu:` 
  Initial implementation using the GPU for gridlet generation.

- `makeGrids4Kernels.cu:` 
  Improved implementation using the GPU for gridlet generation with a better data structure.

- `testBricksOutput.cpp:` 
  Contains tests to compare two given `.grids` files, checking if they contain the same gridlets. Same order of gridlets is not required as they are sorted. If any discrepancies are detected between the provided gridlets, the mismatch details are reported on the console.

- `cubesGeneration.cpp:`
  Generates several sample datasets.

- `mem.sh:`  A tool for monitoring GPU memory usage.

- `timer.h:`  A tool for measuring execution time.

For further information on the rest of the code, please refer to the original [GitHub repository](https://github.com/owl-project/owlExaStitcher) as the rest of the code is left untouched.
## Usage
The code was tested on Ubuntu 22.04 LTS and CUDA version 12.2.
### Build

To build the project follow the instructions below:
```
mkdir build
cd build
mkdir outputGrids
cmake ..
make
```
### Run

To run `makeGrids.cpp` navigate to the `build` folder and provide the path to the `.cubes` file:
```
./amrMakeGrids	        ./path/to/data.cubes
```
you can run the example `denseLevel_0.cubes` file from the `data` folder using:
```
./amrMakeGrids   ../data/denseLevel_0.cubes
```


to run `makeGrids3Kernels.cu`:
```
./amrMakeGrids_cuda3    ./path/to/data.cubes
```
to run `makeGrids4Kernels.cu`:
```
./amrMakeGrids_cuda4    ./path/to/data.cubes
```
### Sample data
Utilize `cubesGeneration.cpp` to create sample data for testing and benchmarking. Compile the code:
```
g++ cubesGeneration.cpp -o gen
```
run it:
```
./gen
```
and follow the instructions.




### Output

The output grids of `makeGrids.cpp`, `makeGrids3Kernels.cu` and `makeGrids4Kernels.cu` are stored in the `build/outputGrids` directory.
