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

#pragma once

#include "umesh/io/IO.h"
#include "umesh/UMesh.h"

namespace umesh {
  namespace io {
    namespace fun3d {
      
      /*! read header from fun3d data file, and reutrn info on what's
          contained */
      void getInfo(const std::string        &dataFileName,
                   std::vector<std::string> &variables,
                   std::vector<int>         &timeSteps);

      /*! read one time step for one variable, from given file; the
          array of global vertex IDs is optional - if set, we'll use
          it to store that data file's vertex Ids that say where these
          scalars are supposed to go in the reconstitued single mesh */
      std::vector<float> readTimeStep(const std::string &dataFileName,
                                      const std::string &desiredVariable,
                                      int desiredTimeStep,
                                      std::vector<uint64_t> *globalVertexIDs=nullptr);
      
    }
  }
}
