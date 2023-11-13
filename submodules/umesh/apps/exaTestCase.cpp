// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
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
#include <fstream>
#include <map>

#ifndef PRINT
#ifdef __CUDA_ARCH__
# define PRINT(va) /**/
# define PING /**/
#else
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif
#endif

namespace umesh {
  std::ofstream cells;
  std::ofstream scalars;

  void grid(vec3i org, vec3i count, int level)
  {
    int cellSize = 1<<level;
    for (int iz=0;iz<count.z;iz++)
      for (int iy=0;iy<count.y;iy++)
        for (int ix=0;ix<count.x;ix++) {
          vec3i pos = org + vec3i(ix,iy,iz)*cellSize;
          float scalar = 1e-5f*length(vec3f(pos));
          std::cout << "writing cell " << pos << ":" << level << " = " << scalar << std::endl;
          cells.write((char*)&pos,sizeof(pos));
          cells.write((char*)&level,sizeof(level));
          scalars.write((char*)&scalar,sizeof(scalar));
        }
  }

  /* one big cell next to 2x2x2 small cells */
  void test1()
  {
    grid(/*pos*/{0,0,0}, /*count(in x,y,z)*/2, /*level*/0);
    grid(/*pos*/{2,0,0}, /*count(in x,y,z)*/1, /*level*/1);
  }

  /* a 2x2x4 pillar of 1 wide, next two a 1x1x2 stack of 2-wide,
     forming a boundary of 2 pyramids and one wedge (plus 3 regular
     hexes) */
  void test2()
  {
    grid(/*pos*/{0,0,0}, /*count(in x,y,z)*/{2,2,4}, /*level*/0);
    grid(/*pos*/{2,0,0}, /*count(in x,y,z)*/{1,1,2}, /*level*/1);
  }

  /* a 4x8x8 wall of 1 wide, next two a 1x2x2 wall of 4-wide */
  void test3()
  {
    grid(/*pos*/{0,0,0}, /*count(in x,y,z)*/{4,8,8}, /*level*/0);
    grid(/*pos*/{4,0,0}, /*count(in x,y,z)*/{1,2,2}, /*level*/2);
  }

  void addOrOverWriteCells(std::map<vec3i,int> &grid,
                           vec3i worldPos,
                           vec3i worldSize,
                           int level)
  {
    int cellWidth = 1<<level;
    for (int iz=0;iz<worldSize.z;iz+=cellWidth)
      for (int iy=0;iy<worldSize.y;iy+=cellWidth)
        for (int ix=0;ix<worldSize.x;ix+=cellWidth) {
          vec3i pos = worldPos+vec3i{ix,iy,iz};
          grid[pos] = level;
        }
  }
  /* 
     something more coplex, with a mix of 4s, 2s, and 1s.
  */
  void test4()
  {
    std::map<vec3i,int> grid; 
    addOrOverWriteCells(grid,{0,0,0},{8,4,4},1);
    addOrOverWriteCells(grid,{0,0,0},{8,2,2},0);

    // addOrOverWriteCells(grid,{0,0,0},{4,4,4},1);
    // addOrOverWriteCells(grid,{0,0,0},{4,2,2},0);
    
    // addOrOverWriteCells(grid,{0,0,0},{8,8,8},1);
    // addOrOverWriteCells(grid,{0,0,0},{8,4,4},0);
    
    // addOrOverWriteCells(grid,{0,8,0},{8,8,16},1);
    // addOrOverWriteCells(grid,{14,0,12},{2,8,4},0);
    // addOrOverWriteCells(grid,{6,8,6},{2,2,8},0);

    for (auto &cell : grid) {
      vec3i pos = cell.first;
      int level = cell.second;
      float scalar = 1e-5f*length(vec3f(pos));
      std::cout << "writing cell " << pos << ":" << level << " = " << scalar << std::endl;
      cells.write((char*)&pos,sizeof(pos));
      cells.write((char*)&level,sizeof(level));
      scalars.write((char*)&scalar,sizeof(scalar));
    }
  }

  void test5()
  {
    std::map<vec3i,int> grid; 
    addOrOverWriteCells(grid,{0,0,0},{16,16,16},2);
    addOrOverWriteCells(grid,{0,0,8},{16,8,8},1);
    addOrOverWriteCells(grid,{0,8,0},{8,8,16},1);
    
    // addOrOverWriteCells(grid,{4,8,4},{4,4,4},0);
    addOrOverWriteCells(grid,{0,0,12},{2,2,2},0);

    for (auto &cell : grid) {
      vec3i pos = cell.first;
      int level = cell.second;
      float scalar = 1e-5f*length(vec3f(pos));
      std::cout << "writing cell " << pos << ":" << level << " = " << scalar << std::endl;
      cells.write((char*)&pos,sizeof(pos));
      cells.write((char*)&level,sizeof(level));
      scalars.write((char*)&scalar,sizeof(scalar));
    }
    PRINT(grid.size());
  }
  
  
  extern "C" int main(int ac, char **av)
  {
    cells.open((std::string)av[1]+".cells",std::ios::binary);
    scalars.open((std::string)av[1]+".scalars",std::ios::binary);

    int testCase = 0;
    if (ac == 3)
      testCase = std::stoi(av[2]);
    switch(testCase) {
    case 1:
      test1();
      break;
    case 2:
      test2();
      break;
    case 3:
      test3();
      break;
    case 4:
      test4();
      break;
    case 5:
      test5();
      break;
    default:
      test1();
    }
  }
}
