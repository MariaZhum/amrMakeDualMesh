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

  /*! tetrahedralize a given input mesh; in a way that non-triangular
    faces (from wedges, pyramids, and hexes) will always be
    subdividied exactly the same way even if that same face is used
    by another elemnt with different vertex order
  
    Notes:

    - tetrahedralization will create new vertices; if the input
    contained a scalar field the output will contain the same
    scalars for existing vertices, and will interpolate for newly
    created ones.
      
    - output contains only tets, and thus will *not* contain the
    input's surface elements (is any such exist)
    
    - the input's vertices will be fully contained within the output's
    vertex array, with the same indices; so any surface elements
    defined in the input should remain valid in the output's vertex
    array.
  */
  UMesh::SP tetrahedralize(UMesh::SP mesh);

  /*! same as tetrahedralize(umesh), but will, eventually, contain
    only the tetrahedra created by the 'owned' elements listed, EVEN
    THOUGH the vertex array produced by that will be the same as in
    the original tetrahedralize(mesh) version */
  UMesh::SP tetrahedralize(UMesh::SP in,
                           int ownedTets,
                           int ownedPyrs,
                           int ownedWedges,
                           int ownedHexes);
  
  /*! same as tetrahedralize, but chop up ONLY elements with curved
      sides, and pass through all those that have flat sides. Note
      this will ALSO (do the best job it can at) flipping
      negative-volume leemnts to positive volume */
  UMesh::SP tetrahedralize_maintainFlatElements(UMesh::SP mesh);
  
} // ::umesh

