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

namespace umesh {

  /*! given a umesh with volumetric elemnets (any sort), compute a new
      umesh (containing only triangles) that contains the triangular
      iso-surface for given iso-value. Input *must* have a per-vertex
      scalar field, but can have any combinatoin of volumetric
      elemnets; tris and quads in the input get ignored; input remains
      unchanged. */
  UMesh::SP extractIsoSurface(UMesh::SP input, float isoValue);
  
} // ::umesh

