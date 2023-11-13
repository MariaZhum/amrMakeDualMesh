// ======================================================================== //
// Copyright 2018-2021 Ingo Wald, 2023 Maria Zhumabaeva                     //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "umesh/UMesh.h"
#include "umesh/io/IO.h"
#include "umesh/check.h"
// #include "tetty/UMesh.h"
#include <cstring>
#include <set>
#include <map>
#include <fstream>
#include <atomic>
#include <array>
#include <chrono>
#include "timer.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iomanip>


#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
#define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
#define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif


using namespace umesh;
using namespace std::chrono;

const int macroCellWidth = 8;
const bool PRINT_EVERY_BRICK_SCALAR = false;
const bool PRINT_STAT = false;

template <typename T>
inline T __host__ __device__ iDivUp(T a, T b){
  return (a+b-1) / b;
}

struct Cube{
  vec3f lower;
  int level;
  std::array<int, 8> scalarIDs;
};

struct Brick{
  Brick(int lvl){
    level = lvl;
  }

  box3i dbg_bounds;

  __device__ void setAttributes(box3i &bounds, int offsetFirst, int offsetLast){
    dbg_bounds = bounds;
    lower = bounds.lower;

    numCubes.x = bounds.upper.x - bounds.lower.x;
    numCubes.y = bounds.upper.y - bounds.lower.y;
    numCubes.z = bounds.upper.z - bounds.lower.z;

    offset = offsetFirst;
    numScalars = offsetLast - offsetFirst;
  }

  vec3i lower;
  int level;
  vec3i numCubes;
  int *scalarIDs;
  int offset;
  int numScalars;
};

vec3i make_vec3i(vec3f v) { return {int(v.x), int(v.y), int(v.z)}; }
vec3f make_vec3f(vec3i v) { return {float(v.x), float(v.y), float(v.z)}; }

vec3i cellID(const Cube &cube){
  vec3i cid = make_vec3i(cube.lower);
  if (cube.lower.x < 0.f)
    cid.x -= ((1 << cube.level) - 1);
  if (cube.lower.y < 0.f)
    cid.y -= ((1 << cube.level) - 1);
  if (cube.lower.z < 0.f)
    cid.z -= ((1 << cube.level) - 1);
  cid = cid / (1 << cube.level);

  // if (cid == vec3i(-1,-1,-1)) {
  //   PING;
  //   PRINT(cube.lower);
  //   PRINT(cid);
  // }
  return cid;
}

box3i cellBounds(const Cube &cube){
  vec3i cell = cellID(cube);
  return {cell, cell + vec3i(1)};
}

vec3i mcID(const Cube &cube){
  vec3i cid = cellID(cube);
  if (cid.x < 0)
    cid.x -= (macroCellWidth - 1);
  if (cid.y < 0)
    cid.y -= (macroCellWidth - 1);
  if (cid.z < 0)
    cid.z -= (macroCellWidth - 1);
  vec3i mcid = cid / macroCellWidth;
  // if (mcid == vec3i(-1,-1,-1)) {
  //   PING;
  //   PRINT(cube.lower);
  //   PRINT(cellID(cube));
  //   PRINT(mcid);
  // }
  return mcid;
}

__device__ void calcMCID(int &mcIDx, int &mcIDy, int &mcIDz, int cellIDx, int cellIDy, int cellIDz){
  mcIDx = cellIDx;
  mcIDy = cellIDy;
  mcIDz = cellIDz;

  if (cellIDx < 0)
    mcIDx -= (macroCellWidth - 1);
  if (cellIDy < 0)
    mcIDy -= (macroCellWidth - 1);
  if (cellIDz < 0)
    mcIDz -= (macroCellWidth - 1);

  mcIDx = mcIDx / macroCellWidth;
  mcIDy = mcIDy / macroCellWidth;
  mcIDz = mcIDz / macroCellWidth;
}

__device__ void calcCellID(int &cellIDx, int &cellIDy, int &cellIDz, vec3f lower, int level){
  cellIDx = (int)lower.x;
  cellIDy = (int)lower.y;
  cellIDz = (int)lower.z;

  if (lower.x < 0.f)
    cellIDx -= ((1 << level) - 1);
  if (lower.y < 0.f)
    cellIDy -= ((1 << level) - 1);
  if (lower.z < 0.f)
    cellIDz -= ((1 << level) - 1);

  cellIDx = cellIDx / (1 << level);
  cellIDy = cellIDy / (1 << level);
  cellIDz = cellIDz / (1 << level);
}

box3f worldBounds(const Brick &brick){
  box3f bb;
  bb.lower = make_vec3f(brick.lower * (1 << brick.level));
  bb.upper = bb.lower + make_vec3f(brick.numCubes * (1 << brick.level));
  return bb;
}

vec3i getLevelSizeInMC(vec3f &levelLower, vec3f &levelUpper, int level){
  Cube minCube;
  Cube maxCube;

  minCube.lower = levelLower;
  minCube.level = level;

  maxCube.lower = levelUpper;
  maxCube.level = level;

  return mcID(maxCube) - mcID(minCube) + vec3i(1);
}

// kernel 1
/* calculates bounds for each macrocell depending on cubes(given by their lower coord.)
   and number of cubes in each macrocell
*/
__global__ void setBoundsAndCubes(vec3f *cubesLower, vec3i levelSizeInMC, vec3i levelLower, int *listOfcubesIDXsforMC,
                                  box3i *mcBounds, int level, int *offsetsCubes, int totalNumOfCubes){

  int cubeNum = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (cubeNum < totalNumOfCubes)
  {
    //printf("cubeNum = %d (kernel1 - inside if ) \n", cubeNum);
    int cellIDx, cellIDy, cellIDz;
    calcCellID(cellIDx, cellIDy, cellIDz, cubesLower[cubeNum], level);

    int mcIDx, mcIDy, mcIDz;
    calcMCID(mcIDx, mcIDy, mcIDz, cellIDx, cellIDy, cellIDz);

    int linearMcIDX = mcIDx-levelLower.x  + (mcIDy-levelLower.y) * levelSizeInMC.x + (mcIDz-levelLower.z) * levelSizeInMC.x * levelSizeInMC.y;
  
    // extend bounds
    // min function is associative
    atomicMin(&mcBounds[linearMcIDX].lower.x, cellIDx);
    atomicMin(&mcBounds[linearMcIDX].lower.y, cellIDy);
    atomicMin(&mcBounds[linearMcIDX].lower.z, cellIDz);

    atomicMax(&mcBounds[linearMcIDX].upper.x, cellIDx + 1);
    atomicMax(&mcBounds[linearMcIDX].upper.y, cellIDy + 1);
    atomicMax(&mcBounds[linearMcIDX].upper.z, cellIDz + 1);
    
    int prevOffsetCubes = atomicAdd(&offsetsCubes[linearMcIDX], 1);
 
    listOfcubesIDXsforMC[linearMcIDX * (macroCellWidth * macroCellWidth * macroCellWidth) + prevOffsetCubes] = cubeNum;
    
  }
}

// kernel 2
/*  calculates the max number of scalars for each macrocell
*/
__global__ void calcMaxNumOfScalars(int numOfMC, box3i *mcBounds, int *maxNumOfScalars){
  int brickNum = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("bricknum = %d", brickNum);
  if (brickNum < numOfMC){
    int x = mcBounds[brickNum].upper.x - mcBounds[brickNum].lower.x;
    int y = mcBounds[brickNum].upper.y - mcBounds[brickNum].lower.y;
    int z = mcBounds[brickNum].upper.z - mcBounds[brickNum].lower.z;
    maxNumOfScalars[brickNum] = (x+1) * (y+1) * (z+1);
  }
}

/*  writes scalars of each cube into one array 
    index within array is determined by linear brick number and the number 
    of scalars in previous bricks
*/
__device__ void writeCube(vec3f &cubeLower, Brick &brick, int *scalars, int offset, int *resultScalarArray){
  int cellIDx, cellIDy, cellIDz;

  calcCellID(cellIDx, cellIDy, cellIDz, cubeLower, brick.level);
  // printf("cellID (%d, %d, %d)\n", cellIDx, cellIDy, cellIDz);
  int baseX = cellIDx - brick.lower.x;
  int baseY = cellIDy - brick.lower.y;
  int baseZ = cellIDz - brick.lower.z;

  int vtkOrder[8] = {0, 1, 3, 2, 4, 5, 7, 6};

  // index within brickx, worldSizeInMC.y, worldSizeInMC.z
  int idx;

  for (int iz = 0; iz < 2; iz++)
    for (int iy = 0; iy < 2; iy++)
      for (int ix = 0; ix < 2; ix++){
        idx = baseX + ix + (brick.numCubes.x + 1) * (baseY + iy + (brick.numCubes.y + 1) * (baseZ + iz));
        resultScalarArray[idx + offset] = scalars[vtkOrder[4 * iz + 2 * iy + ix]];
      }
}

/* sets scalars for a given brick (=start position in resultScalarsArr)
   to -1 == empty cell
*/
__device__ void setResultScalarArrToEmpty(int *resultScalarsArr, int start, int end){
  for (int i = start; i <= end; i++){
    resultScalarsArr[i] = -1;
  }
}

// kernel 3
/*  creates brick by writing its attributes 
    (brick is empty if brick.numCubes == 0)
    and writes scalars of that brick into resultScalarsArr
*/
__global__ void createAndFillBricks(vec3f *cubesLower, box3i *mcBounds, int *listOfcubesIDXsforMC, int *scalars, Brick *mcBricks,
                                int level, int *offsetCubes, int totalNumOfMC, int *offsetScalars, int *resultScalarsArr){
  
  int brickNum = blockIdx.x * blockDim.x + threadIdx.x;

  if (brickNum < totalNumOfMC && offsetCubes[brickNum]!=0){
    //set all scalars to empty = -1
    setResultScalarArrToEmpty(resultScalarsArr, offsetScalars[brickNum], offsetScalars[brickNum+1] - 1);

    mcBricks[brickNum].setAttributes(mcBounds[brickNum], offsetScalars[brickNum], offsetScalars[brickNum+1]);

    int scalarsCube[8];
    int cubeidx;

     // for each cube in MC write its scalars into resultScalarsArr
    for (int i = 0; i < offsetCubes[brickNum]; i++){
      cubeidx = listOfcubesIDXsforMC[brickNum * macroCellWidth * macroCellWidth * macroCellWidth + i];

      #pragma unroll
      for (int j = 0; j < 8; j++){
        scalarsCube[j] = scalars[cubeidx * 8 + j];
      }

      writeCube(cubesLower[cubeidx], mcBricks[brickNum], scalarsCube, offsetScalars[brickNum], resultScalarsArr);
    }    
  }
}

// writes statistics for one level
void writeStat(int level, int numOfBricks, int numOfCubes, float kernel1, float kernel2, float kernel3, float totalKernelTime, float thrustPrefSum, float step1, float step2, float step3, float totalStepTime){
  std::ofstream outFile;
  outFile.open ("stat.txt", std::ofstream::out | std::ofstream::app);

  if(!outFile.is_open()){
    std::cout << "Error opening file!" << std::endl;
    return;
  }

  outFile << "level = " << level << ", number of generated cubes = " << numOfCubes-1 << ", number of generated bricks = " << numOfBricks << std::endl;

  outFile << "+" << std::setfill('-') << std::setw(20) << "+" << std::setw(24) << "+" << std::setw(18) << "+" << std::endl;
  outFile << std::left << std::setfill(' ') << std::setw(20)<< "|" << "|" << std::setw(23) << "GPU (incl. alloc/cpy)" << "|" << std::setw(17) << "GPU (kernel only)" <<"|" << std::endl;
  outFile << std::setfill('-') << std::setw(20) << "+"  << std::setw(24) << "+"  << std::setw(18) << "+" << "+" << std::endl;

  outFile << std::left << "|" << std::setfill(' ') << std::setw(19) << "step/kernel 1" << "|" << std::setw(23) << step1 << "|" << std::setw(17) << kernel1 << "|"<< std::endl;
  outFile << std::left << "|" << std::setfill(' ') << std::setw(19) << "step/kernel 2" << "|" << std::setw(23) <<  step2 << "|" << std::setw(17) << kernel2 << "|"<< std::endl;
  outFile << std::left << "|" << std::setfill(' ') << std::setw(19) << "thrust" << std::setw(24) << "|" << "|" << std::setw(17) << thrustPrefSum<< "|"<< std::endl;
  outFile << std::left << "|" << std::setfill(' ') << std::setw(19) << "step/kernel 3"<< "|" << std::setw(23) << step3 << "|" << std::setw(17) << kernel3 << "|"<< std::endl;
  outFile << std::left << "|" << std::setfill(' ') << std::setw(19) << "total"<< "|" << std::setw(23) << totalStepTime<< "|" << std::setw(17) << totalKernelTime << "|"<< std::endl;

  outFile << std::setfill('-') << std::setw(20) << "+"  << std::setw(24) << "+"  << std::setw(18) << "+" << "+" << std::endl;

  outFile << " " << std::endl;
  outFile << " " << std::endl;

  outFile.close();

}

/*! the 'cells' are all in a space where each cell is exactly 1
    int-coord wide, so the second cell on level 1 is _not_ at
    (2,2,2)-(4,4,4), but at (1,1,1)-(2,2,2). To translate from this
    level-L cell space to world coordinates, take cell (i,j,k) and get
    lower=((i,j,k)+.5f)*(1<<L), and upper = lower+(1<<L) */
std::vector<Brick> makeBricksForLevel(int level,
                                      std::vector<Cube> &cubes, int *&resultScalarArray){
  auto start = high_resolution_clock::now();

  gridlets::timer t;

  size_t numOfCubes = cubes.size();

  // lower and upper .lower point of cubes for current lvl in world coord 
  vec3f levelLower = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
  vec3f levelUpper = vec3f(FLT_MIN, FLT_MIN, FLT_MIN);

  std::vector<vec3f> cubesLower;
  cubesLower.reserve(numOfCubes);

  std::vector<int> scalarsArray;
  scalarsArray.reserve(8*numOfCubes);

  t.reset();

  for (auto cube: cubes){
    levelLower = min(levelLower, cube.lower);
    levelUpper = max(levelUpper, cube.lower);
    cubesLower.push_back(cube.lower);
    for (int j = 0; j < 8; j++){
      scalarsArray.push_back(cube.scalarIDs[j]);
    }
  }

  std::cout << __LINE__ << " " << t.elapsed() << "s time for copying .lower and .scalars into arrays and finding min/max\n"
            << std::endl;
  t.reset();

 //for shifting MC Grid to the point of origin since we use [0] as the starting index and not the actual macrocell id 
  Cube lowestCube;
  lowestCube.lower = levelLower;
  lowestCube.level = level;

  // size of grid in mc determined by cubes
  vec3i levelSizeInMC = getLevelSizeInMC(levelLower, levelUpper, level);

  size_t numberOfMC = levelSizeInMC.x * levelSizeInMC.y * levelSizeInMC.z;
  
  t.reset();

  // 1st kernel
  std::vector<box3i> mcBounds;
  mcBounds.resize(numberOfMC);

  // alloc mem device
  box3i *ptr_mcBounds;
  int *ptr_listOfcubesIDXsforMC;
  vec3f *ptr_cubesLower;
  int *ptr_offsetsCubes;

  t.reset();
  cudaDeviceSynchronize();
  std::cout << __LINE__ << " " << t.elapsed() << "s time for setting up cuda \n"
            << std::endl;
  t.reset();

  cudaMalloc((void **)&ptr_mcBounds, numberOfMC * sizeof(box3i));
  cudaMalloc((void **)&ptr_listOfcubesIDXsforMC, numberOfMC * (macroCellWidth * macroCellWidth * macroCellWidth) * sizeof(int));
  cudaMalloc((void **)&ptr_cubesLower, numOfCubes * sizeof(vec3f));
  cudaMalloc((void **)&ptr_offsetsCubes, numberOfMC * sizeof(int));
  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 1 alloc. \n"
            << std::endl;
  t.reset();

  cudaMemcpy(ptr_mcBounds, &mcBounds[0], numberOfMC * sizeof(box3i), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_cubesLower, &cubesLower[0], numOfCubes * sizeof(vec3f), cudaMemcpyHostToDevice);
  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 1 copy\n"
            << std::endl;
  t.reset();

  size_t numThreads = 1024;

  setBoundsAndCubes<<<iDivUp(numOfCubes, numThreads), numThreads>>>(ptr_cubesLower, levelSizeInMC, mcID(lowestCube),
                                                                    ptr_listOfcubesIDXsforMC, ptr_mcBounds, level, ptr_offsetsCubes, numOfCubes);
  cudaPeekAtLastError();


  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 1 run time\n"
            << std::endl;
  float kernel1Time = t.elapsed();
  t.reset();

  auto timeAfterFirstStep = high_resolution_clock::now();

  // 2nd kernel

  // offsets for scalars
  int *ptr_maxNumOfScalars;

  // last entry = size of maxScalarsArray
  cudaMalloc((void **)&ptr_maxNumOfScalars, (numberOfMC + 1) * sizeof(int));
  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 2 alloc. \n"
            << std::endl;
  t.reset();

  calcMaxNumOfScalars<<<iDivUp(numberOfMC, numThreads), numThreads>>>(numberOfMC, ptr_mcBounds, ptr_maxNumOfScalars);

  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 2 run time\n"
            << std::endl;

  float kernel2Time = t.elapsed();
  t.reset();

  thrust::device_ptr<int> thr_ptr_maxNumOfScalars = thrust::device_pointer_cast(ptr_maxNumOfScalars);

  thrust::exclusive_scan(thr_ptr_maxNumOfScalars, thr_ptr_maxNumOfScalars + numberOfMC + 1, thr_ptr_maxNumOfScalars);
  ptr_maxNumOfScalars = thrust::raw_pointer_cast(thr_ptr_maxNumOfScalars);

  int totalNumberOfScalars;
  cudaMemcpy(&totalNumberOfScalars, ptr_maxNumOfScalars + numberOfMC, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << __LINE__ << " " << t.elapsed() << "s prefixsum run time\n"
            << std::endl;
  float prefixSumTime = t.elapsed();
  t.reset();

  auto timeAfterSecondStep = high_resolution_clock::now();

  // 3rd kernel
  Brick *ptr_mcBricks;
  int *ptr_scalarsArray;
  int *ptr_resultScalarsArray;

  resultScalarArray = new int[totalNumberOfScalars];

  std::vector<Brick> mcBricks(numberOfMC, level);
  mcBricks.reserve(numberOfMC);

  cudaMalloc((void **)&ptr_scalarsArray, numOfCubes * 8 * sizeof(int));
  cudaMalloc((void **)&ptr_mcBricks, numberOfMC * sizeof(Brick));
  cudaMalloc((void **)&ptr_resultScalarsArray, totalNumberOfScalars * sizeof(int));
  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 3 alloc. \n"
            << std::endl;
  t.reset();

  cudaMemcpy(ptr_scalarsArray, &scalarsArray[0], 8 * numOfCubes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_mcBricks, &mcBricks[0], numberOfMC * sizeof(Brick), cudaMemcpyHostToDevice);
  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 3 copy\n"
            << std::endl;
  t.reset();

  createAndFillBricks<<<iDivUp(numberOfMC, numThreads), numThreads>>>(ptr_cubesLower, ptr_mcBounds, ptr_listOfcubesIDXsforMC, ptr_scalarsArray, 
                                                                  ptr_mcBricks, level, ptr_offsetsCubes, numberOfMC, ptr_maxNumOfScalars, ptr_resultScalarsArray);
  

  std::cout << __LINE__ << " " << t.elapsed() << "s kernel 3 run time\n" << std::endl;
  float kernel3Time= t.elapsed();
  t.reset();

  cudaMemcpy(&mcBricks[0], ptr_mcBricks, numberOfMC * sizeof(Brick), cudaMemcpyDeviceToHost);
  cudaMemcpy(&resultScalarArray[0], ptr_resultScalarsArray, totalNumberOfScalars * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << __LINE__ << " " << t.elapsed() << "s copy result to CPU \n" << std::endl;
  t.reset();

  cudaFree(ptr_cubesLower);
  cudaFree(ptr_mcBounds);
  cudaFree(ptr_listOfcubesIDXsforMC);
  cudaFree(ptr_mcBricks);
  cudaFree(ptr_scalarsArray);
  cudaFree(ptr_offsetsCubes);
  cudaFree(ptr_maxNumOfScalars);
  cudaFree(ptr_resultScalarsArray);

  // set pointers: Brick-> resultScalarArray
  for (size_t i = 0; i < numberOfMC; i++){
    mcBricks[i].scalarIDs = &resultScalarArray[mcBricks[i].offset];
  }

  auto timeAfterThirdStep = high_resolution_clock::now();

  if (PRINT_EVERY_BRICK_SCALAR){
    int brickNum = 0;
    int brickOffset = mcBricks[0].offset;
    printf("Scalars: ");
    for (int i = 0; i < totalNumberOfScalars; i++){
      if (i == brickOffset){
        // new brick beginns
        std::cout << "\n printing scalars for brick.lower = :" << mcBricks[brickNum].lower << std::endl;
        brickNum++;
        brickOffset = mcBricks[brickNum].offset;
      }
      printf("%d ", resultScalarArray[i]);
    }
    printf("\n---------------------------\n");
  }

  float timeForStep1 = (timeAfterFirstStep - start).count() / 1000000000.0; 
  float timeForStep2 = (timeAfterSecondStep - timeAfterFirstStep).count() / 1000000000.0;
  float timeForStep3 = (timeAfterThirdStep- timeAfterSecondStep).count() / 1000000000.0; 
  float totalStepTime = (timeAfterThirdStep - start).count() / 1000000000.0;

  std::cout << "Time taken by first step: "
            << timeForStep1 << " s" << std::endl;

  std::cout << "Time taken by second step: "
            << timeForStep2 << " s" << std::endl;

  std::cout << "Time taken by third step: "
            << timeForStep3 << " s" << std::endl;

  std::cout << "Time taken by entire function: "
            << totalStepTime << " s" << std::endl;

  if (PRINT_STAT){
    writeStat(level, numberOfMC, numOfCubes, kernel1Time, kernel2Time, kernel3Time, kernel1Time+kernel2Time+kernel3Time+prefixSumTime, 
                prefixSumTime, timeForStep1, timeForStep2, timeForStep3, totalStepTime);
  }

  return mcBricks;
}

void writeQuadOBJ(std::ostream &out,
                  vec3f base,
                  vec3f du,
                  vec3f dv){
  vec3f v00 = base;
  vec3f v01 = base + du;
  vec3f v11 = base + du + dv;
  vec3f v10 = base + dv;
  out << "v " << v00.x << " " << v00.y << " " << v00.z << std::endl;
  out << "v " << v01.x << " " << v01.y << " " << v01.z << std::endl;
  out << "v " << v10.x << " " << v10.y << " " << v10.z << std::endl;
  out << "v " << v11.x << " " << v11.y << " " << v11.z << std::endl;
  out << "f -1 -2 -4 -3" << std::endl;
}

void writeOBJ(std::ostream &out, const box3f &box){
  vec3f dx(box.size().x, 0.f, 0.f);
  vec3f dy(0.f, box.size().y, 0.f);
  vec3f dz(0.f, 0.f, box.size().z);
  writeQuadOBJ(out, box.lower, dx, dy);
  writeQuadOBJ(out, box.lower, dx, dz);
  writeQuadOBJ(out, box.lower, dy, dz);
  writeQuadOBJ(out, box.upper, -dx, -dy);
  writeQuadOBJ(out, box.upper, -dx, -dz);
  writeQuadOBJ(out, box.upper, -dy, -dz);
}

void writeBIN(std::ostream &out, const Brick &brick){
  out.write((const char *)&brick.lower, sizeof(brick.lower));
  out.write((const char *)&brick.level, sizeof(brick.level));
  out.write((const char *)&brick.numCubes, sizeof(brick.numCubes));
  out.write((const char *)brick.scalarIDs, brick.numScalars * sizeof(int));
}

void makeGridsFor(const std::string &fileName){
  std::cout << "==================================================================" << std::endl;
  std::cout << "making grids for " << fileName << std::endl;
  std::cout << "==================================================================" << std::endl;
  const char *ext = strstr(fileName.c_str(), "_");
  if (!ext)
    throw std::runtime_error("'" + fileName + "' is not a cubes file!?");
  while (const char *next = strstr(ext + 1, "_"))
    ext = next;
  int level;
  int rc = sscanf(ext, "_%i.cubes", &level);
  if (rc != 1)
    throw std::runtime_error("'" + fileName + "' is not a cubes file!?");
  
  std::vector<Cube> cubes;
  std::ifstream in(fileName, std::ios::binary);
  
  gridlets::timer t2;
  while (!in.eof())
  {
    Cube cube;
    in.read((char *)&cube, sizeof(cube));
    cubes.push_back(cube);
  }

  std::cout << t2.elapsed() << "s for copying cubes from file" << std::endl;

  // scalars for each brick are stored here consecutively
  int *resultScalarArray = NULL;

  std::vector<Brick> bricks = makeBricksForLevel(level, cubes, resultScalarArray);

#if 1
  int numBricksGenerated = 0;
  int numCubesInBricks = 0;
  int numScalarsInBricks = 0;
  for (auto &brick : bricks){
    if(brick.numCubes.x != 0){
      numBricksGenerated++;
      numCubesInBricks += brick.numCubes.x * brick.numCubes.y * brick.numCubes.z;
      numScalarsInBricks += brick.numScalars;
    }
  }
  PRINT(numBricksGenerated);
  PRINT(numCubesInBricks);
  PRINT(numScalarsInBricks);
  static int totalBricksGenerated = 0;
  static int totalCubesInBricks = 0;
  static int totalScalarsInBricks = 0;

  totalBricksGenerated += numBricksGenerated;
  totalCubesInBricks += numCubesInBricks;
  totalScalarsInBricks += numScalarsInBricks;

  PRINT(totalBricksGenerated);
  PRINT(totalCubesInBricks);
  PRINT(totalScalarsInBricks);
  PRINT(prettyNumber(totalBricksGenerated));
  PRINT(prettyNumber(totalCubesInBricks));
  PRINT(prettyNumber(totalScalarsInBricks));

  static int fileID = 0;
  std::ofstream out;
  
  std::string outName = "./outputGrids/cuda_k3_level_"+std::to_string(level)+".grids";

  if (fileID++ == 0)
    out.open(outName, std::ios_base::binary);
  else
    out.open(outName, std::ios_base::binary | std::ios_base::app);
  for (auto &brick : bricks){
    if(brick.numCubes.x != 0){
      writeBIN(out, brick);
    }
  }
#else
  std::ofstream out("./outputGrids/out.obj");
  for (auto &brick : bricks)
  {
    writeOBJ(out, worldBounds(brick));
  }
#endif

  delete[] resultScalarArray;
}

int main(int ac, char **av){
  gridlets::timer t_sum;

  if(PRINT_STAT){
    std::ofstream outFile;
    outFile.open ("stat.txt", std::ofstream::out | std::ofstream::app);

    if(outFile.is_open()){
      outFile << __FILE__ << std::endl;
    }
    else{
      std::cout << "Error opening file!" << std::endl;
    }

    for (int i = 1; i < ac; i++){
      outFile << av[i] << std::endl; 
      makeGridsFor(av[i]);
    }    
  }
  else{
    for (int i = 1; i < ac; i++){
      makeGridsFor(av[i]);
    }
  }
    
  std::cout << t_sum.elapsed() << "s for all levels" << std::endl; 
  t_sum.reset();
}
