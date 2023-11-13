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

#include "exabin.h"
#include <fstream>
#include <map>

namespace tetty {
  namespace io {


    size_t getOrAddVertex(vec3i pos, std::map<vec3i,size_t> &mapping, UMesh::SP umesh)
    {
      auto it = mapping.find(pos);
      if (it != mapping.end()) return it->second;

      size_t newID = umesh->vertex.size();
      assert(vec3i(vec3f(pos)) == pos);
      umesh->vertex.push_back(vec3f(pos));
      mapping[pos] = newID;
      return newID;
    }

    UMesh::SP importExaBin(const std::string &hexFileName,
                           const std::string &scalarFileName)
    {
      std::ifstream hexFile(hexFileName);
      if (!hexFile.is_open()) 
        throw std::runtime_error(std::string("Unable to open file " + hexFileName));
      assert(hexFile.good());
      
      std::ifstream scalarFile(scalarFileName);
      assert(scalarFile.good());
      if (!scalarFile.is_open()) 
        throw std::runtime_error(std::string("Unable to open file " + scalarFileName));
      
      UMesh::SP result = std::make_shared<UMesh>();
      result->perHex = std::make_shared<Attribute>();
      // result->perHex->value.push_back((float)inScalar);

      std::map<vec3i,size_t> vertexID;
      
      struct ExaHex {
        vec3i pos;
        int   level;
      };
      int nextPing = 1;
      while (!hexFile.eof()) {
        assert(!scalarFile.eof());
        ExaHex inHex;
        float inScalar;

        assert(hexFile.good());
        hexFile.read((char*)&inHex,sizeof(inHex));
        assert(scalarFile.good());
        scalarFile.read((char*)&inScalar,sizeof(inScalar));
        int width = 1 << inHex.level;

        UMesh::Hex hex;
        hex.base.x = (int)getOrAddVertex(inHex.pos+vec3i(0,0,0)*width,vertexID,result);
        hex.base.y = (int)getOrAddVertex(inHex.pos+vec3i(0,0,1)*width,vertexID,result);
        hex.base.z = (int)getOrAddVertex(inHex.pos+vec3i(0,1,1)*width,vertexID,result);
        hex.base.w = (int)getOrAddVertex(inHex.pos+vec3i(0,1,0)*width,vertexID,result);
        hex.top.x = (int)getOrAddVertex(inHex.pos+vec3i(1,0,0)*width,vertexID,result);
        hex.top.y = (int)getOrAddVertex(inHex.pos+vec3i(1,0,1)*width,vertexID,result);
        hex.top.z = (int)getOrAddVertex(inHex.pos+vec3i(1,1,1)*width,vertexID,result);
        hex.top.w = (int)getOrAddVertex(inHex.pos+vec3i(1,1,0)*width,vertexID,result);

        result->hexes.push_back(hex);
        result->perHex->value.push_back((float)inScalar);

        if (result->hexes.size() >= nextPing) {
          std::cout << "#exabin: now have " << prettyNumber(result->hexes.size()) << " hexes ..." << std::endl;
          nextPing += nextPing;
        }
      }
      result->finalize();
      return result;
    }

  } // ::tetty::io
} // ::tetty
