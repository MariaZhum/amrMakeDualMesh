// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

/* computes the outer shell of a tet-mesh, ie, all the triangle and/or
   bilinear faces that are _not_ shared between two neighboring
   elements */ 

#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/io/btm/BTM.h"
#include "umesh/RemeshHelper.h"
#include <algorithm>

namespace umesh {

  int numSwaps_tets = 0;
  int numSwaps_pyrs = 0;
  int numSwaps_wedges = 0;
  int numSwaps_hexes = 0;

  void ping()
  {
    printf("\rnum swaps: t=%i, p=%i, w=%i, h=%i\t\t",
           numSwaps_tets,
           numSwaps_pyrs,
           numSwaps_wedges,
           numSwaps_hexes);
  }
  
  void swappingTet()
  {
    numSwaps_tets++; ping();
  }
  void swappingPyr()
  {
    numSwaps_pyrs++; ping();
  }
  void swappingWedge()
  {
    numSwaps_wedges++; ping();
  }
  void swappingHex()
  {
    numSwaps_hexes++; ping();
  }

  inline float volume(const vec3f &v0,
                      const vec3f &v1,
                      const vec3f &v2,
                      const vec3f &v3)
  {
    return dot(v3-v0,cross(v1-v0,v2-v0));
  }
                      
  void fixup(UMesh::SP mesh, UMesh::Tet &tet)
  {
    vec3f v0 = mesh->vertices[tet[0]];
    vec3f v1 = mesh->vertices[tet[1]];
    vec3f v2 = mesh->vertices[tet[2]];
    vec3f v3 = mesh->vertices[tet[3]];

    if (volume(v0,v1,v2,v3) < 0.f) {
      std::swap(tet.x,tet.y);
      swappingTet();
    }
  }
  
  void fixup(UMesh::SP mesh, UMesh::Pyr &pyr)
  {
    vec3f v0 = mesh->vertices[pyr[0]];
    vec3f v1 = mesh->vertices[pyr[1]];
    vec3f v2 = mesh->vertices[pyr[2]];
    vec3f v3 = mesh->vertices[pyr[3]];
    vec3f v4 = mesh->vertices[pyr[4]];

    vec3f b = 0.25f*(v0+v1+v2+v3);
    if (volume(v0,v1,b,v4) < 0.f) {
      std::swap(pyr.base.x,pyr.base.y);
      std::swap(pyr.base.z,pyr.base.w);
      swappingPyr();
    }
  }
  
  void fixup(UMesh::SP mesh, UMesh::Wedge &wedge)
  {
    vec3f v0 = mesh->vertices[wedge[0]];
    vec3f v1 = mesh->vertices[wedge[1]];
    vec3f v2 = mesh->vertices[wedge[2]];
    vec3f v3 = mesh->vertices[wedge[3]];
    vec3f v4 = mesh->vertices[wedge[4]];
    vec3f v5 = mesh->vertices[wedge[5]];

    vec3f b = 0.25f*(v0+v1+v3+v4);
    if (volume(v3,v4,v5,b) < 0.f) {
      std::swap(wedge[0],wedge[3]);
      std::swap(wedge[1],wedge[4]);
      std::swap(wedge[2],wedge[5]);
      swappingWedge();
    }
  }

  void fixup(UMesh::SP mesh, UMesh::Hex &hex)
  {
    vec3f v0 = mesh->vertices[hex[0]];
    vec3f v1 = mesh->vertices[hex[1]];
    vec3f v2 = mesh->vertices[hex[2]];
    vec3f v3 = mesh->vertices[hex[3]];
    vec3f v4 = mesh->vertices[hex[4]];
    vec3f v5 = mesh->vertices[hex[5]];
    vec3f v6 = mesh->vertices[hex[6]];
    vec3f v7 = mesh->vertices[hex[7]];

    vec3f c = 0.125f*(v0+v1+v2+v3+v4+v5+v6+v7);
    vec3f b = 0.25f*(v0+v1+v2+v3);
    if (volume(v0,v1,b,c) < 0.f) {
      std::swap(hex[0],hex[4]);
      std::swap(hex[1],hex[5]);
      std::swap(hex[2],hex[6]);
      std::swap(hex[3],hex[7]);
      swappingHex();
    }
  }
  
  void fixNegativeVolumes(UMesh::SP mesh)
  {
    for (auto &tet   : mesh->tets)   fixup(mesh,tet);
    for (auto &pyr   : mesh->pyrs)   fixup(mesh,pyr);
    for (auto &wedge : mesh->wedges) fixup(mesh,wedge);
    for (auto &hex   : mesh->hexes)  fixup(mesh,hex);
  }
  
  extern "C" int main(int ac, char **av)
  {
    try {
      std::string inFileName;
      std::string outFileName;

      for (int i = 1; i < ac; i++) {
        const std::string arg = av[i];
        if (arg == "-o")
          outFileName = av[++i];
        else if (arg[0] != '-')
          inFileName = arg;
        else {
          throw std::runtime_error("./umeshComputeShell <in.umesh> -o <out.shell> [-tribin file.tribin] [--obj <out.obj]");
        }
      }
      if (outFileName == "")
        throw std::runtime_error("no output filename specified (-o)");

      std::cout << "loading umesh from " << inFileName << std::endl;
      UMesh::SP in = io::loadBinaryUMesh(inFileName);
      std::cout << "flipping negative elements ....\n\n";
      fixNegativeVolumes(in);

      io::saveBinaryUMesh(outFileName, in);
      std::cout << "done saving umesh file" << std::endl;
    }
    catch (std::exception &e) {
      std::cerr << "fatal error " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }  
} // ::umesh
