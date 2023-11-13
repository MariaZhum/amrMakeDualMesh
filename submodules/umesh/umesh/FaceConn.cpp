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

#include "FaceConn.h"
#include "umesh/io/IO.h"

# ifdef UMESH_HAVE_TBB
#  include "tbb/parallel_sort.h"
# endif
#include <set>
#include <algorithm>
#include <string.h>
#include <fstream>

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

  using std::swap;
  
  using SharedFace   = FaceConn::SharedFace;
  using PrimFacetRef = FaceConn::PrimFacetRef;

  std::ostream &operator<<(std::ostream &out, const PrimFacetRef &ref)
  {
    out << "Ref{type="<<ref.primType<<",fct="<<ref.facetIdx<<",primID="<<ref.primIdx<<"}";
    return out;
  }
  
  /*! one "face" of a given prim, defined through the (global) vertex
      indices, the gloabl primitive index, and the (local to the prim)
      facet index; orientation defines whether the given vertex
      indices point inwards, or outwards, allowing to later keeping
      track of which side of a face the given facet belons */
  struct Facet {
    vec4i        vertexIdx;
    PrimFacetRef prim;
    int          orientation;
  };

  std::ostream &operator<<(std::ostream &out, const Facet &facet)
  {
    out << "Facet{vtx="<<facet.vertexIdx<<",prim="<<facet.prim<<",orientation="<<facet.orientation<<"}";
    return out;
  }

  /*! allows for sorting facets in a unqieu order based on their
      indices, but _ignoring_ things like orientation; this allows to
      later find facets belonding to the same face by first
      re-ordering the facet's vertex indices ina unque way and then
      sorting by that order (which means two facets for the same face
      shold then end up next to each other after sorting) */
  struct FacetComparator {
    inline 
    bool operator()(const Facet &a, const Facet &b) const {
      return
        (a.vertexIdx.x < b.vertexIdx.x)
        ||
        ((a.vertexIdx.x == b.vertexIdx.x) &&
         (a.vertexIdx.y <  b.vertexIdx.y))
        ||
        ((a.vertexIdx.x == b.vertexIdx.x) &&
         (a.vertexIdx.y == b.vertexIdx.y) &&
         (a.vertexIdx.z <  b.vertexIdx.z))
        ||
        ((a.vertexIdx.x == b.vertexIdx.x) &&
         (a.vertexIdx.y == b.vertexIdx.y) &&
         (a.vertexIdx.z == b.vertexIdx.z) &&
         (a.vertexIdx.w <  b.vertexIdx.w));
    }
  };

               
  /*! describes input data through plain pointers, so we can run the
    same algorithm once with std::vector::data() (on the host) or
    cuda-malloced data (on gpu) */
  struct InputMesh {
    Tet   *tets;
    size_t numTets;
    Pyr   *pyrs;
    size_t numPyrs;
    Wedge *wedges;
    size_t numWedges;
    Hex   *hexes;
    size_t numHexes;
  };

  inline int numUniqueVertices(vec3i v)
  {
    std::sort(&v.x,&v.x+3);
    int cnt = 1;
    for (int i=1;i<3;i++)
      if (v[i] != v[i-1]) cnt++;
    return cnt;
  }
  inline int numUniqueVertices(vec4i v)
  {
    std::sort(&v.x,&v.x+4);
    int cnt = 1;
    for (int i=1;i<4;i++)
      if (v[i] != v[i-1]) cnt++;
    return cnt;
  }
  
  // ==================================================================
  // compute vertex order stage
  // ==================================================================
  
  /*! computes a unique vertex order such that two different prims
    that share the same face - but have written thie facets with
    differnetly ordered vertex indices - will end up with the same
    vector of indices. of course, re-ordeirng indices can change
    orientation, which this function keeps track of */
  inline 
  void computeUniqueVertexOrder(Facet &facet)
  {
    vec4i idx = facet.vertexIdx;
    // \todo optimize this - all we need is detect the number of
    // unique vertices, for which this is overkill
    // std::set<int> uniqueIDs;
    // uniqueIDs.insert(idx.x);
    // uniqueIDs.insert(idx.y);
    // uniqueIDs.insert(idx.z);
    
    if (idx.w < 0) {
      int numUnique = numUniqueVertices((const vec3i&)idx);
      if (numUnique < 3) {
        facet.vertexIdx = vec4i(-1);
        return;
      }
      if (idx.y < idx.x)
        { swap(idx.x,idx.y); facet.orientation = 1-facet.orientation; }
      if (idx.z < idx.x)
        { swap(idx.x,idx.z); facet.orientation = 1-facet.orientation; }
      if (idx.z < idx.y)
        { swap(idx.y,idx.z); facet.orientation = 1-facet.orientation; }
    } else {
      // uniqueIDs.insert(idx.w);
      int numUnique = numUniqueVertices(idx);
      
      if (numUnique == 2) {
        facet.vertexIdx = vec4i(-1);
        return;
      }
      
      if (numUnique == 3) {
        if (idx.x==idx.y) {
          idx = { idx.x, idx.z, idx.w, -1 };
        } else if (idx.x == idx.z) {
          // oooooh... this one is fishy
          PING;
          idx = { idx.x, idx.y, idx.w, -1 };
        } else if (idx.x == idx.w) {
          idx = { idx.x, idx.y, idx.z, -1 };
        } else if (idx.y == idx.z) {
          idx = { idx.x, idx.y, idx.w, -1 };
        } else if (idx.y == idx.w) {
          // oooooh... this one is fishy
          PING;
          idx = { idx.x, idx.y, idx.z, -1 };
        } else if (idx.z == idx.w) {
          idx = { idx.x, idx.y, idx.z, -1 };
        } else throw std::runtime_error("???");
        
        if (idx.y < idx.x)
          { swap(idx.x,idx.y); facet.orientation = 1-facet.orientation; }
        if (idx.z < idx.x)
          { swap(idx.x,idx.z); facet.orientation = 1-facet.orientation; }
        if (idx.z < idx.y)
          { swap(idx.y,idx.z); facet.orientation = 1-facet.orientation; }
      } else {

        int lv = idx.x, li=0;
        if (idx.y < lv) { lv = idx.y; li = 1; }
        if (idx.z < lv) { lv = idx.z; li = 2; }
        if (idx.w < lv) { lv = idx.w; li = 3; }

        switch (li) {
        case 0: idx = { idx.x,idx.y,idx.z,idx.w }; break;
        case 1: idx = { idx.y,idx.z,idx.w,idx.x }; break;
        case 2: idx = { idx.z,idx.w,idx.x,idx.y }; break;
        case 3: idx = { idx.w,idx.x,idx.y,idx.z }; break;
        };

        if (idx.w < idx.y) {
          facet.orientation = 1-facet.orientation;
          swap(idx.w,idx.y);
        }
      }
    }
    facet.vertexIdx = idx;
  }

  void computeUniqueVertexOrder(Facet *facets, size_t numFacets)
  {
    parallel_for_blocked
      (0,numFacets,1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++) {
           computeUniqueVertexOrder(facets[i]);
         }
       });
  }

  // ==================================================================
  // init faces
  // ==================================================================
  
  /*! writes the four facets of a tet; for degenerate tets that may
    end up with faces that collapse to points or lines - that's OK,
    as it'll be fixed later on in computeUniqueVertexOrder() */
  inline 
  void writeTetFacets(Facet *facets,
                      size_t tetIdx,
                      InputMesh mesh
                      )
  {
    for (int i=0;i<4;i++) facets[i].prim.primType = UMesh::TET;
    for (int i=0;i<4;i++) facets[i].prim.facetIdx = i;
    for (int i=0;i<4;i++) facets[i].prim.primIdx  = tetIdx;
    for (int i=0;i<4;i++) facets[i].orientation   = 0;

    vec4i tet = mesh.tets[tetIdx];

    facets[0].vertexIdx = { tet.y,tet.w,tet.z,-1 };
    facets[1].vertexIdx = { tet.x,tet.z,tet.w,-1 };
    facets[2].vertexIdx = { tet.x,tet.w,tet.y,-1 };
    facets[3].vertexIdx = { tet.x,tet.y,tet.z,-1 };
  }
  
  /*! writes the five facets of a pyrs; for degenerate pyramids thta
    may end up with quads that become triangles, and/or entire faces
    that collapse to points or lines - that's OK, as it'll be fixed
    later on in computeUniqueVertexOrder() */
  inline 
  void writePyrFacets(Facet *facets,
                      size_t pyrIdx,
                      InputMesh mesh
                      )
  {
    for (int i=0;i<5;i++) facets[i].prim.primType = UMesh::PYR;
    for (int i=0;i<5;i++) facets[i].prim.facetIdx = i;
    for (int i=0;i<5;i++) facets[i].prim.primIdx  = pyrIdx;
    for (int i=0;i<5;i++) facets[i].orientation   = 0;
    
    UMesh::Pyr pyr = mesh.pyrs[pyrIdx];
    vec4i base = pyr.base;
    facets[0].vertexIdx = { pyr.top,base.y,base.x,-1 };
    facets[1].vertexIdx = { pyr.top,base.z,base.y,-1 };
    facets[2].vertexIdx = { pyr.top,base.w,base.z,-1 };
    facets[3].vertexIdx = { pyr.top,base.x,base.w,-1 };
    facets[4].vertexIdx = { base.x,base.y,base.z,base.w };
  }

  /*! writes the five facets of a wedge; for degenerate wedges thta
    may end up with quads that become triangles, and/or entire faces
    that collapse to points or lines - that's OK, as it'll be fixed
    later on in computeUniqueVertexOrder() */
  inline 
  void writeWedgeFacets(Facet *facets,
                        size_t wedgeIdx,
                        InputMesh mesh
                        )
  {
    for (int i=0;i<5;i++) facets[i].prim.primType = UMesh::WEDGE;
    for (int i=0;i<5;i++) facets[i].prim.facetIdx = i;
    for (int i=0;i<5;i++) facets[i].prim.primIdx  = wedgeIdx;
    for (int i=0;i<5;i++) facets[i].orientation   = 0;
    
    UMesh::Wedge wedge = mesh.wedges[wedgeIdx];
    int i0 = wedge.front.x;
    int i1 = wedge.front.y;
    int i2 = wedge.front.z;
    int i3 = wedge.back.x;
    int i4 = wedge.back.y;
    int i5 = wedge.back.z;

    facets[0].vertexIdx = { i0,i2,i1,-1 };
    facets[1].vertexIdx = { i3,i4,i5,-1 };
    facets[2].vertexIdx = { i0,i3,i5,i2 };
    facets[3].vertexIdx = { i1,i2,i5,i4 };
    facets[4].vertexIdx = { i0,i1,i4,i3 };
  }
  
  /*! writes the five facets of a hexes; for degenerate hexes that may
    end up with quads that become triangles, and/or entire faces
    that collapse to points or lines - that's OK, as it'll be fixed
    later on in computeUniqueVertexOrder() */
  inline 
  void writeHexFacets(Facet *facets,
                      size_t hexIdx,
                      InputMesh mesh
                      )
  {
    for (int i=0;i<6;i++) facets[i].prim.primType = UMesh::HEX;
    for (int i=0;i<6;i++) facets[i].prim.facetIdx = i;
    for (int i=0;i<6;i++) facets[i].prim.primIdx  = hexIdx;
    for (int i=0;i<6;i++) facets[i].orientation   = 0;
    
    UMesh::Hex hex = mesh.hexes[hexIdx];
    int i0 = hex.base.x;
    int i1 = hex.base.y;
    int i2 = hex.base.z;
    int i3 = hex.base.w;
    int i4 = hex.top.x;
    int i5 = hex.top.y;
    int i6 = hex.top.z;
    int i7 = hex.top.w;

    facets[0].vertexIdx = { i0,i1,i2,i3 };
    facets[1].vertexIdx = { i4,i7,i6,i5 };
    facets[2].vertexIdx = { i0,i4,i5,i1 };
    facets[3].vertexIdx = { i2,i6,i7,i3 };
    facets[4].vertexIdx = { i1,i5,i6,i2 };
    facets[5].vertexIdx = { i0,i3,i7,i4 };
  }

  /*! writes the facets given by the parallel launch that runs over
    all facets in the model */
  inline 
  void writeFacets(Facet *facets, size_t jobIdx, const InputMesh &mesh)
  {
    // write tets
    if (jobIdx < mesh.numTets) {
      writeTetFacets(facets+4*jobIdx,jobIdx,mesh);
      return;
    }
    facets += 4*mesh.numTets;
    jobIdx -= mesh.numTets;
  
    // write pyramids
    if (jobIdx < mesh.numPyrs) {
      writePyrFacets(facets+5*jobIdx,jobIdx,mesh);
      return;
    }
    facets += 5*mesh.numPyrs;
    jobIdx -= mesh.numPyrs;
  
    // write wedges
    if (jobIdx < mesh.numWedges) {
      writeWedgeFacets(facets+5*jobIdx,jobIdx,mesh);
      return;
    }
    facets += 5*mesh.numWedges;
    jobIdx -= mesh.numWedges;
  
    // write hexes
    if (jobIdx < mesh.numHexes) {
      writeHexFacets(facets+6*jobIdx,jobIdx,mesh);
      return;
    }
    return;
  }

  /*! writes ALL facets in the mesh, even degenerate ones */
  void writeFacets(Facet *facets,
                   const InputMesh &mesh)
  {
    size_t numPrims
      = mesh.numTets
      + mesh.numPyrs
      + mesh.numWedges
      + mesh.numHexes;
    parallel_for_blocked
      (0,numPrims,1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++)
           writeFacets(facets,i,mesh);
       });
  }

  // ==================================================================
  // sort facet array
  // ==================================================================
  void sortFacets(Facet *facets, size_t numFacets)
  {
# ifdef UMESH_HAVE_TBB
    std::cout << "parallel face sorting" << std::endl;
    tbb::parallel_sort(facets,facets+numFacets,FacetComparator());
    std::cout << "done face sorting" << std::endl;
# else
    std::sort(facets,facets+numFacets,FacetComparator());
# endif
  }
  
  // ==================================================================
  // set up / upload input mesh data
  // ==================================================================
  template<typename T>
  inline void upload(T *&ptr, size_t &count,
                     const std::vector<T> &vec)
  {
    ptr = (T*)vec.data();
    count = vec.size();
  }

  void setupInput(InputMesh &mesh, UMesh::SP input)
  {
    upload(mesh.tets,mesh.numTets,input->tets);
    upload(mesh.pyrs,mesh.numPyrs,input->pyrs);
    upload(mesh.wedges,mesh.numWedges,input->wedges);
    upload(mesh.hexes,mesh.numHexes,input->hexes);
  }
  

  // ==================================================================
  // let facets write the facess
  // ==================================================================
  inline 
  void facetsWriteFacesKernel(SharedFace *faces,
                              const Facet *facets,
                              const uint64_t *faceIndices,
                              size_t facetIdx)
  {
    const Facet facet = facets[facetIdx];

    if (facet.vertexIdx.x < 0)
      return;
    
    assert(faceIndices[facetIdx] > 0);
    size_t faceIdx = faceIndices[facetIdx]-1;
    SharedFace &face = faces[faceIdx];
    auto &side = facet.orientation ? face.onFront : face.onBack;
    face.vertexIdx = facet.vertexIdx;
    if (!(side.primIdx < 0)) {
      // iw: this sanity check is not thread safe
      PRINT(facet);
      PRINT(facetIdx);
      PRINT(faceIdx);
      PRINT(face.onFront);
      PRINT(face.onBack);
      PRINT(side);
      throw std::runtime_error("side is used twice!?");
    }
    side = facet.prim;
  }
  
  void facetsWriteFaces(SharedFace *faces,
                        const Facet *facets,
                        const uint64_t *faceIndices,
                        size_t numFacets)
  {
    parallel_for_blocked
      (0,numFacets,1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++)
           facetsWriteFacesKernel(faces,facets,faceIndices,i);
       });
  }
  
  // ==================================================================
  // compute face indices from (sorted) facet array
  // ==================================================================

  inline 
  void initFaceIndexKernel(uint64_t *faceIndices,
                           Facet *facets,
                           size_t facetIdx)
  {
    faceIndices[facetIdx]
      =  (facetIdx == 0)
      || (facets[facetIdx-1].vertexIdx.x != facets[facetIdx].vertexIdx.x)
      || (facets[facetIdx-1].vertexIdx.y != facets[facetIdx].vertexIdx.y)
      || (facets[facetIdx-1].vertexIdx.z != facets[facetIdx].vertexIdx.z)
      || (facets[facetIdx-1].vertexIdx.w != facets[facetIdx].vertexIdx.w);
  }

  void clearFaces(SharedFace *faces, size_t numFaces)
  {
    PrimFacetRef clearPrim = { 0,0,-1 };
    clearPrim.primIdx = -1;
    for (size_t i=0;i<numFaces;i++) {
      faces[i].onFront = faces[i].onBack = clearPrim;
      faces[i].vertexIdx = vec4i(-1);
    }
  }
  
  void initFaceIndices(uint64_t *faceIndices,
                       Facet *facets,
                       size_t numFacets)
  {
    parallel_for_blocked
      (0,numFacets,1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++)
           initFaceIndexKernel(faceIndices,facets,i);
       });
    
    //  for (size_t i=0;i<numFacets;i++) {
    // initFaceIndexKernel(faceIndices,facets,i);
    // }
  }
  
  /*! not parallelized... this will likely be mem bound, anyway */
  void prefixSum(uint64_t *faceIndices,
                 size_t numFacets)
  {
    size_t sum = 0;
    for (size_t i=0;i<numFacets;i++) {
      size_t old = faceIndices[i];
      faceIndices[i] = sum;
      sum += old;
    }
  }
  void postfixSum(uint64_t *faceIndices,
                 size_t numFacets)
  {
    size_t sum = 0;
    for (size_t i=0;i<numFacets;i++) {
      size_t old = faceIndices[i];
      sum += old;
      faceIndices[i] = sum;
    }
  }


  // ==================================================================
  // aaaand ... wrap it all together
  // ==================================================================

std::vector<SharedFace> computeFaces(UMesh::SP input)
  {
    assert(input);
    std::chrono::steady_clock::time_point
      begin_inc = std::chrono::steady_clock::now();
    InputMesh mesh;
    setupInput(mesh,input);

    std::chrono::steady_clock::time_point
      begin_exc = std::chrono::steady_clock::now();

    // -------------------------------------------------------
    size_t numFacets
      = 4 * mesh.numTets
      + 5 * mesh.numPyrs
      + 5 * mesh.numWedges
      + 6 * mesh.numHexes;
    if (numFacets == 0)
      /*! this is critical: without this we'll be getting a crash
          later on when it tries to access the "last" facet */
      return {};
    
    std::vector<Facet> facets(numFacets);
    writeFacets(facets.data(),mesh);
    // for (int i=0;i<numFacets;i++)
    //   std::cout << "facet " << i << " = " << facets[i] << std::endl;
    computeUniqueVertexOrder(facets.data(),numFacets);

    // -------------------------------------------------------
    sortFacets(facets.data(),numFacets);
    std::vector<uint64_t> faceIndices(numFacets);
    initFaceIndices(faceIndices.data(),facets.data(),numFacets);
    // prefixSum(faceIndices,numFacets);
    postfixSum(faceIndices.data(),numFacets);
    // -------------------------------------------------------
    size_t numFaces = faceIndices[numFacets-1];
    std::vector<SharedFace> faces(numFaces);
    // SharedFace *faces = result.data();//allocateFaces(result,numFaces);
    clearFaces(faces.data(),numFaces);
    
    // -------------------------------------------------------
    facetsWriteFaces(faces.data(),facets.data(),faceIndices.data(),numFacets);
    
    // finishFaces(result,faces,numFaces);
    // freeIndices(faceIndices);
    // freeFacets(facets);
    return faces;
  }

  /*! given a unstructured mesh, compute the face-connectivity for
    this mesh. Note this _sohuld_ work even for curved/bilinear faces,
    but will error out for meshes with bad connectivyt (faces with
    more than two owning prims) */
  FaceConn::SP FaceConn::compute(UMesh::SP input)
  {
    FaceConn::SP faceConn = std::make_shared<FaceConn>();
    faceConn->faces = computeFaces(input);
    return faceConn;
  }

  /*! write - binary - to given file */
  void FaceConn::write(std::ostream &out) const
  {
    io::writeVector(out,faces);
  }
  
  /*! read from given file, assuming file format as used by saveTo() */
  void FaceConn::read(std::istream &in)
  {
    io::readVector(in,faces);
  }

  /*! write - binary - to given file */
  void FaceConn::saveTo(const std::string &fileName) const
  {
    std::ofstream out(fileName,std::ios::binary);
    write(out);
  }

  /*! read from given file, assuming file format as used by saveTo() */
  FaceConn::SP FaceConn::loadFrom(const std::string &fileName)
  {
    FaceConn::SP conn = std::make_shared<FaceConn>();
    std::ifstream in(fileName,std::ios::binary);
    conn->read(in);
    return conn;
  }
    
} // ::umesh
