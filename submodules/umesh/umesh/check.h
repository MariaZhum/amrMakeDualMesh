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

#pragma once

#include "UMesh.h"

#ifdef NDEBUG
#else
#  define UMESH_ENABLE_SANITY_CHECKS 0///
#endif
namespace umesh {

  /* if specified, the sanity checker will ignore 'no volume prims' */
#define CHECK_FLAG_MESH_IS_SURFACE (1<<0)

#if UMESH_ENABLE_SANITY_CHECKS
  /*! perform some sanity checking of the given mesh (checking indices
    are valid, etc) */
  void sanityCheck(UMesh::SP umesh, uint32_t flags = 0);
#else
  inline void sanityCheck(UMesh::SP umesh, uint32_t flags = 0)
  { /* sanity checks disabled by flag */ }
#endif
  
} // :: umesh
