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

#include "bin.h"

namespace tetty {
  namespace io {

    UMesh::SP importNateBIN(const std::string &fileName)
    {
      std::ifstream file;
      file.open(fileName, std::ios::in | std::ios::binary );
      if (!file.is_open()) 
        throw std::runtime_error(std::string("Unable to open file " + fileName));
      
      UMesh::SP result = std::make_shared<UMesh>();

      uint32_t points_per_primitive;
      readElement(file,points_per_primitive);
      
      uint32_t num_points;
      readElement(file,num_points);
      
      uint32_t num_indices;
      readElement(file,num_indices);

      uint8_t data_is_per_cell;
      readElement(file,data_is_per_cell);

      result->vertex.resize(num_points);
      readArray(file,result->vertex.data(),num_points);

      if (data_is_per_cell) {
        int numScalars = num_indices / points_per_primitive;
        result->perTet = std::make_shared<Attribute>(numScalars);
        readArray(file,result->perTet->value.data(),result->perTet->value.size());
      } else {
        int numScalars = num_points;
        result->perVertex = std::make_shared<Attribute>(numScalars);
        readArray(file,result->perVertex->value.data(),result->perVertex->value.size());
      }
      
      result->tets.resize(num_indices/points_per_primitive);
      readArray(file,result->tets.data(),result->tets.size());

      result->finalize();
      return result;
    }
    

  } // ::tetty::io
} // ::tetty
