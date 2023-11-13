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

#include "UMesh.h"

namespace umesh {
  /*! helper clas that allows to create a new umesh's vertex array
    from the vertices of one or more other meshes, allowing to find
    the duplicates and translating the input meshes' vertices to the
    new mesh's vertex indices */
  struct RemeshHelper {
    RemeshHelper(UMesh &target);

    /*! find ID of given vertex in target mesh (if it already exists),a
      nd return it; otherwise add vertex to target mesh, and return
      new ID */
    uint32_t getID(const vec3f &v);
    
    /*! given a vertex v, return its ID in the target mesh's vertex
      array (if present), or add it (if not). To afterwards allow the
      using libnray to look up which of the inptu vertices ended up
      where we also allow to specify a vertex "tag" for each input
      vertices. meshes should only ever get built with *either* this
      functoin of the one that uses a float scalar, not mixed */
    uint32_t getID(const vec3f &v, size_t tag);

    /*! given a vertex v and associated per-vertex scalar value s,
      return its ID in the target mesh's vertex array (if present), or
      add it (if not). To afterwards allow the using libnray to look
      up which of the inptu vertices ended up where we also allow to
      specify a vertex "tag" for each input vertices.  meshes should
      only ever get built with *either* this functoin of the one that
      uses a size_t tag, not mixed */
    uint32_t getID(const vec3f &v, float scalar);

    /*! given a vertex ID in another mesh, return an ID for the
        *output* mesh that corresponds to this vertex (add to output
        if not already present) */
    uint32_t translate(const uint32_t in,
                       UMesh::SP otherMesh);

    /*! translate one set of vertex indices from the old mesh's
        indexing to our new indexing */
    void translate(uint32_t *indices, int N,
                   UMesh::SP otherMesh);
    /*! translate one set of vertex indices from the old mesh's
        indexing to our new indexing */
    void translate(int32_t *indices, int N,
                   UMesh::SP otherMesh)
    { translate((uint32_t*)indices,N,otherMesh); }

    void add(UMesh::SP otherMesh, UMesh::PrimRef primRef);
    
    std::map<vec3f,uint32_t> knownVertices;
    
    UMesh &target;
    // std::vector<size_t> vertexTag;
  };
  
  void removeDuplicatesAndUnusedVertices(UMesh::SP mesh);

  /*! removed all vertices that are not used by any prim, and
      re-indexes all prims with the new vertex/value array indices
      after this compaction. CAREFUL: this function assumes that the mesh does
      NOT have duplicate vertices */
  void removeUnusedVertices(UMesh::SP mesh);

} // ::tetty
