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

#include "umesh/extractSurfaceMesh.h"
#include "umesh/RemeshHelper.h"

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

  /*! Given a umesh with possibly mixed surface and volumetric
      eements, create a new umesh that contains _only_ surface
      elements, and, in particular, only vertices used by these
      surface elements.
  */
  UMesh::SP extractSurfaceMesh(UMesh::SP input)
  {
    UMesh::SP output = std::make_shared<UMesh>();
    RemeshHelper helper(*output);

    for (auto prim : input->createSurfacePrimRefs())
      helper.add(input,prim);
    return output;
  }
} // ::umesh

