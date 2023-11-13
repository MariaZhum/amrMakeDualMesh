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

#include "umesh/UMesh.h"
#include "umesh/io/IO.h"

namespace umesh {
  namespace io {

    /*! loader for the specially modified fun3d format that uses 64-bit indices */
    struct UGrid64Loader {
      UGrid64Loader(const std::string &dataFileName,
                    const std::string &scalarFileName);

      static UMesh::SP load(const std::string &dataFileName,
                            const std::string &scalarFileName="");
      
      UMesh::SP result; 
    };

  }
}
