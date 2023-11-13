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

#include "RemeshHelper.h"
# ifdef UMESH_HAVE_TBB
#  include "tbb/parallel_sort.h"
# endif

namespace umesh {

  RemeshHelper::RemeshHelper(UMesh &target)
    : target(target)
  {}

  /*! given a vertex v, return its ID in the target mesh's vertex
    array (if present), or add it (if not). To afterwards allow the
    using libnray to look up which of the inptu vertices ended up
    where we also allow to specify a vertex "tag" for each input
    vertices. meshes should only ever get built with *either* this
    functoin of the one that uses a float scalar, not mixed */
   uint32_t RemeshHelper::getID(const vec3f &v, size_t tag)
  {
    auto it = knownVertices.find(v);
    if (it != knownVertices.end()) {
      return it->second;
    }
    int ID = (int)target.vertices.size();
    knownVertices[v] = ID;
    target.vertexTag.push_back(tag);
    target.vertices.push_back(v);
    return ID;
  }

  /*! find ID of given vertex in target mesh (if it already exists),a
    nd return it; otherwise add vertex to target mesh, and return
    new ID */
   uint32_t RemeshHelper::getID(const vec3f &v)
  {
    assert(!target.perVertex);
    auto it = knownVertices.find(v);
    if (it != knownVertices.end()) {
      return it->second;
    }
    int ID = (int)target.vertices.size();
    knownVertices[v] = ID;
    target.vertices.push_back(v);
    return ID;
  }
  
  /*! given a vertex v and associated per-vertex scalar value s,
    return its ID in the target mesh's vertex array (if present), or
    add it (if not). To afterwards allow the using libnray to look
    up which of the inptu vertices ended up where we also allow to
    specify a vertex "tag" for each input vertices.  meshes should
    only ever get built with *either* this functoin of the one that
    uses a size_t tag, not mixed */
   uint32_t RemeshHelper::getID(const vec3f &v, float scalar)
  {
    auto it = knownVertices.find(v);
    if (it != knownVertices.end()) {
      return it->second;
    }
    int ID = (int)target.vertices.size();
    knownVertices[v] = ID;
    if (!target.perVertex)
      target.perVertex = std::make_shared<Attribute>();
    target.perVertex->values.push_back(scalar);
    target.vertices.push_back(v);
    return ID;
  }

  /*! given a vertex ID in another mesh, return an ID for the
   *output* mesh that corresponds to this vertex (add to output
   if not already present) */
   uint32_t RemeshHelper::translate(const uint32_t in,
                                   UMesh::SP otherMesh)
  {
    assert(otherMesh);
    assert(in >= 0);
    assert(in < otherMesh->vertices.size());
    if (otherMesh->perVertex) {
      return getID(otherMesh->vertices[in],
                   otherMesh->perVertex->values[in]);
    } else if (!otherMesh->vertexTag.empty()) {
      return getID(otherMesh->vertices[in],
                   otherMesh->vertexTag[in]);
    } else {
      if (target.perVertex)
        throw std::runtime_error("can't translate a vertex from another mesh that has neither scalars not vertex tags");
      else
        return getID(otherMesh->vertices[in]);
    }
  }
  
  void RemeshHelper::translate(uint32_t *indices, int N,
                               UMesh::SP otherMesh)
  {
    assert(otherMesh);
    for (int i=0;i<N;i++)
      indices[i] = translate(indices[i],otherMesh);
  }

  bool noDuplicates(const Triangle &tri)
  {
    return
      tri.x != tri.y &&
      tri.x != tri.z &&
      tri.y != tri.z;
  }
  
  bool noDuplicates(const Tet &tet)
  {
    return
      tet.x != tet.y &&
      tet.x != tet.z &&
      tet.x != tet.w &&
      tet.y != tet.z &&
      tet.y != tet.w &&
      tet.z != tet.w;
  }
  
  void RemeshHelper::add(UMesh::SP otherMesh, UMesh::PrimRef primRef)
  {
    switch (primRef.type) {
    case UMesh::TRI: {
      auto prim = otherMesh->triangles[primRef.ID];
      translate((uint32_t*)&prim,3,otherMesh);
      if (noDuplicates(prim))
        target.triangles.push_back(prim);
    } break;
    case UMesh::QUAD: {
      auto prim = otherMesh->quads[primRef.ID];
      translate((uint32_t*)&prim,4,otherMesh);
      target.quads.push_back(prim);
    } break;

    case UMesh::TET: {
      auto prim = otherMesh->tets[primRef.ID];
      translate((uint32_t*)&prim,4,otherMesh);
      if (noDuplicates(prim))
        target.tets.push_back(prim);
    } break;
    case UMesh::PYR: {
      auto prim = otherMesh->pyrs[primRef.ID];
      translate((uint32_t*)&prim,5,otherMesh);
      target.pyrs.push_back(prim);
    } break;
    case UMesh::WEDGE: {
      auto prim = otherMesh->wedges[primRef.ID];
      translate((uint32_t*)&prim,6,otherMesh);
      target.wedges.push_back(prim);
    } break;
    case UMesh::HEX: {
      auto prim = otherMesh->hexes[primRef.ID];
      translate((uint32_t*)&prim,8,otherMesh);
      target.hexes.push_back(prim);
    } break;
    default:
      throw std::runtime_error("un-implemented prim type?");
    }
  }




  struct BigVertex {
    vec3f pos;
    float scalar;
    uint32_t orgID;
    int active;;
  };

  inline bool operator<(const BigVertex &a,
                        const BigVertex &b)
  {
    return a.pos < b.pos;
  }
  
  void removeDuplicatesAndUnusedVertices(UMesh::SP mesh)
  {
    std::cout << "parallel reindexing : init for " << mesh->toString() << std::endl;
    std::vector<BigVertex> vertices(mesh->vertices.size());

    // generate list of _all_ vertices, in 'fat' layout that can easily be re-ordered
    parallel_for_blocked
      (0,vertices.size(),16*1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++) {
           vertices[i].pos = mesh->vertices[i];
           vertices[i].scalar = mesh->perVertex->values[i];
           vertices[i].orgID = i;
           vertices[i].active = false;
         }
       });

    // mark 
    for (auto &prim : mesh->triangles) {
      for (int i=0;i<prim.numVertices;i++)
        vertices[prim[i]].active = true;
    }
    for (auto &prim : mesh->quads) {
      for (int i=0;i<prim.numVertices;i++)
        vertices[prim[i]].active = true;
    }
    parallel_for_blocked
      (0,mesh->tets.size(),1024*64,
       [&](size_t begin, size_t end){
        for (size_t primID=begin;primID<end;primID++){
          auto &prim = mesh->tets[primID];
          for (int i=0;i<prim.numVertices;i++)
            vertices[prim[i]].active = true;
        }});
    for (auto &prim : mesh->pyrs) {
      for (int i=0;i<prim.numVertices;i++)
        vertices[prim[i]].active = true;
    }
    for (auto &prim : mesh->wedges) {
      for (int i=0;i<prim.numVertices;i++)
        vertices[prim[i]].active = true;
    }
    for (auto &prim : mesh->hexes) {
      for (int i=0;i<prim.numVertices;i++)
        vertices[prim[i]].active = true;
    }




    


    
    std::cout << "parallel reindexing - sorting vertices to find duplicates" << std::endl;
# ifdef UMESH_HAVE_TBB
    tbb::parallel_sort(vertices.data(),vertices.data()+vertices.size());
# else
    std::sort(vertices.data(),vertices.data()+vertices.size());
# endif

    std::cout << "parallel reindexing - finding unique used vertices" << std::endl;
    int curID = -1;
    std::vector<int> newID(vertices.size());
    for (size_t i=0;i<vertices.size();i++) {
      if (!vertices[i].active) {
        newID[vertices[i].orgID] = -1;
        continue;
      }
      
      if (i==0 || vertices[i].pos != vertices[i-1].pos)
        ++curID;
      
      vertices[curID].pos = vertices[i].pos;
      vertices[curID].scalar = vertices[i].scalar;
      newID[vertices[i].orgID] = curID;
    }
    int numNewVertices = curID;
    std::cout << "num vertices found : " << numNewVertices << std::endl;
    mesh->vertices.resize(numNewVertices);
    mesh->perVertex->values.resize(numNewVertices);
    for (int i=0;i<numNewVertices;i++) {
      mesh->vertices[i] = vertices[i].pos;
      mesh->perVertex->values[i] = vertices[i].scalar;
    }
    vertices.clear();

    std::cout << "parallel reindexing - translating indices" << std::endl;
    for (auto &prim : mesh->triangles) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->quads) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    parallel_for_blocked
      (0,mesh->tets.size(),1024*64,
       [&](size_t begin, size_t end){
        for (size_t primID=begin;primID<end;primID++){
          auto &prim = mesh->tets[primID];
          for (int i=0;i<prim.numVertices;i++)
            prim[i] = newID[prim[i]];
        }});
    for (auto &prim : mesh->pyrs) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->wedges) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->hexes) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
  }

  void removeUnusedVertices(UMesh::SP mesh)
  {
    std::vector<uint8_t> isUsed(mesh->vertices.size());
    for (auto &flag : isUsed) flag = false;
    
    // mark 
    for (auto &prim : mesh->triangles) {
      for (int i=0;i<prim.numVertices;i++)
        isUsed[prim[i]] = true;
    }
    for (auto &prim : mesh->quads) {
      for (int i=0;i<prim.numVertices;i++)
        isUsed[prim[i]] = true;
    }
    parallel_for_blocked
      (0,mesh->tets.size(),1024*64,
       [&](size_t begin, size_t end){
        for (size_t primID=begin;primID<end;primID++){
          auto &prim = mesh->tets[primID];
          for (int i=0;i<prim.numVertices;i++)
            isUsed[prim[i]] = true;
        }});
    for (auto &prim : mesh->pyrs) {
      for (int i=0;i<prim.numVertices;i++)
        isUsed[prim[i]] = true;
    }
    for (auto &prim : mesh->wedges) {
      for (int i=0;i<prim.numVertices;i++)
        isUsed[prim[i]] = true;
    }
    for (auto &prim : mesh->hexes) {
      for (int i=0;i<prim.numVertices;i++)
        isUsed[prim[i]] = true;
    }

    std::vector<int> newID(mesh->vertices.size());
    int curID = 0;
    for (int i=0;i<newID.size();i++) {
      if (isUsed[i]) {
        mesh->vertices[curID] = mesh->vertices[i];
        if (mesh->perVertex)
          mesh->perVertex->values[curID] = mesh->perVertex->values[i];
        newID[i] = curID++;
      } else {
        // won't get used, anyway....
        newID[i] = -1;
      }
    }
    std::cout << "done compacting vertex array, num vertices found " << curID << std::endl;
    mesh->vertices.resize(curID);
    mesh->perVertex->values.resize(curID);

    
    for (auto &prim : mesh->triangles) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->quads) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    parallel_for_blocked
      (0,mesh->tets.size(),1024*64,
       [&](size_t begin, size_t end){
        for (size_t primID=begin;primID<end;primID++){
          auto &prim = mesh->tets[primID];
          for (int i=0;i<prim.numVertices;i++)
            prim[i] = newID[prim[i]];
        }});
    for (auto &prim : mesh->pyrs) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->wedges) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
    for (auto &prim : mesh->hexes) {
      for (int i=0;i<prim.numVertices;i++)
        prim[i] = newID[prim[i]];
    }
  }
  
} // ::umesh
