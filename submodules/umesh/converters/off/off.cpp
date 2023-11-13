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

#include "tetty/io/off/off.h"

namespace tetty {
  namespace io {

    UMesh::SP importOFF(const std::string &fileName)
    {
      std::ifstream in(fileName);
      if (!in.good())
        throw io::CouldNotOpenException(fileName);

      UMesh::SP result = std::make_shared<UMesh>();
      result->perVertex = std::make_shared<Attribute>();
      
      int numVerts, numTets;
      in >> numVerts >> numTets;
      std::cout << "#tetty.io.off: going to read "
                << prettyNumber(numTets) << " with "
                << prettyNumber(numVerts) << " vertices" << std::endl;
      
      // ==================================================================
      for (int i=0;i<numVerts;i++) {
        vec3f pos;
        float val;
        in >> pos.x;
        in >> pos.y;
        in >> pos.z;
        in >> val;
        assert(in.good());

        result->vertex.push_back(pos);
        result->bounds.extend(pos);
        result->perVertex->value.push_back(val);
      }

      // ==================================================================
      for (int i=0;i<numTets;i++) {
        UMesh::Tet tet;
        in >> tet.x;
        in >> tet.y;
        in >> tet.z;
        in >> tet.w;
        assert(in.good());

        box3f bounds;
        bounds.extend(result->vertex[tet.x]);
        bounds.extend(result->vertex[tet.y]);
        bounds.extend(result->vertex[tet.z]);
        bounds.extend(result->vertex[tet.w]);
        if (bounds.lower == bounds.upper)
          continue;

        result->tets.push_back(tet);
        result->bounds.extend(bounds);
      }

      // ==================================================================
      result->perVertex->finalize();
      return result;
    }

  } // ::tetty::io
} // ::tetty
