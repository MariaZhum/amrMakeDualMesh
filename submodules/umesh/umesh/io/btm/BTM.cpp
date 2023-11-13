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

#include "BTM.h"
#include "umesh/io/IO.h"
#include <fstream>

namespace umesh {
  namespace btm {

    static Mesh::SP load(const std::string &fileName)
    {
      Mesh::SP mesh = std::make_shared<Mesh>();
      std::ifstream in(fileName,std::ios::binary);
      mesh->loadFrom(in);
      return mesh;
    }
      
    void Mesh::loadFrom(std::ifstream &in)
    {
      io::readVector(in,vertex);
      io::readVector(in,normal);
      io::readVector(in,color);
      io::readVector(in,texcoord);
      io::readVector(in,index);
      io::readVector(in,triColor);
    }
      
    void Mesh::save(const std::string &fileName) const
    {
      std::ofstream out(fileName,std::ios::binary);
      saveTo(out);
    }
      
    void Mesh::saveTo(std::ofstream &out) const
    {
      io::writeVector(out,vertex);
      io::writeVector(out,normal);
      io::writeVector(out,color);
      io::writeVector(out,texcoord);
      io::writeVector(out,index);
      io::writeVector(out,triColor);
    }
      
  } // ::umesh::btm
} // ::umesh
