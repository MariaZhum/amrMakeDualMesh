// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#include "TestCase.h"

namespace tetty {
  namespace io {

    inline int vertexID(int N, int x, int y, int z)
    {
      return x + (N+1)*(y+ (N+1)*(z));
    }
    
    UMesh::SP createTestCase(const std::string &dataFileName)
    {
      UMesh::SP mesh = std::make_shared<UMesh>();
      int N = atoi(dataFileName.c_str());
      mesh->perVertex = std::make_shared<Attribute>();

      for (int iz=0;iz<=N;iz++)
        for (int iy=0;iy<=N;iy++)
          for (int ix=0;ix<=N;ix++) {
            vec3f v(vec3i(ix,iy,iz));
            mesh->vertex.push_back(v);
            mesh->perVertex->value.push_back(length(v));
          }
      mesh->perVertex->finalize();

      for (int iz=0;iz<N;iz++)
        for (int iy=0;iy<N;iy++)
          for (int ix=0;ix<N;ix++) {
            UMesh::Hex hex;
            hex.base.x = vertexID(N,ix+0,iy+0,iz+0);
            hex.base.y = vertexID(N,ix+1,iy+0,iz+0);
            hex.base.z = vertexID(N,ix+1,iy+1,iz+0);
            hex.base.w = vertexID(N,ix+0,iy+1,iz+0);

            hex.top.x = vertexID(N,ix+0,iy+0,iz+1);
            hex.top.y = vertexID(N,ix+0,iy+1,iz+1);
            hex.top.z = vertexID(N,ix+1,iy+1,iz+1);
            hex.top.w = vertexID(N,ix+1,iy+0,iz+1);

            mesh->hexes.push_back(hex);
          }
      mesh->finalize();
      return mesh;
    }
      
  } // ::tetty::io
} // ::tetty

