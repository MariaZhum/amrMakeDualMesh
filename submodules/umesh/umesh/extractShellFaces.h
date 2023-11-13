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

  /*! given a umesh with mixed volumetric elements, create a a new
    mesh of surface elemnts (ie, triangles and quads) that
    corresponds to the outside facing "shell" faces of the input
    elements (ie, all those that re not shared by two different
    elements. All surface elements in the output mesh will be
    OUTWARD facing. */
  UMesh::SP extractShellFaces(UMesh::SP mesh,
                              /*! if true, we'll create a new set of
                                vertices for ONLY the required
                                vertices. If false, we'll leave the
                                vertices array empty, and have the
                                vertex indices refer to the
                                original input mesh */
                              bool remeshVertices);
} // ::umesh

