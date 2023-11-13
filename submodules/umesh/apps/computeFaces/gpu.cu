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

#include "umesh/UMesh.h"
#ifdef __CUDACC__
# include <thrust/sort.h>
#endif
# ifdef UMESH_HAVE_TBB
#  include "tbb/parallel_sort.h"
# endif
#include <set>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <string.h>

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

#define CUDA_CHECK( call )                                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUDA_CALL(call) CUDA_CHECK(cuda##call)

#define CUDA_CHECK2( where, call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUDA_SYNC_CHECK()                                       \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }

namespace umesh {
  
#ifndef __CUDA_ARCH__
  typedef vec4i int4;
  using std::swap;
#else
  template<typename T>
  inline __umesh_both__ void swap(T &a, T &b)
  {
    T c = a; a = b; b = c;
  }
  
#endif
  
  struct PrimFacetRef{
    /*! only 4 possible types (tet, pyr, wedge, or hex) */
    uint64_t primType:3;
    /*! only 8 possible facet it could be (in a hex) */
    uint64_t facetIdx:3;
    int64_t  primIdx:58;
  };
  
  struct UMESH_ALIGN(16) Facet {
    int4         vertexIdx;
    PrimFacetRef prim;
    int          orientation;
  };
  
  struct FacetComparator {
    inline __umesh_both__
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


  struct SharedFace {
    int4 vertexIdx;
    PrimFacetRef onFront, onBack;
  };
               
  void usage(const std::string &error)
  {
    if (error == "") std::cout << "Error: " << error << "\n\n";
    std::cout << "Usage: ./umeshComputeFaces{CPU|GPU} in.umesh -o out.faces\n";
    exit(error != "");
  }

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

  // ==================================================================
  // compute vertex order stage
  // ==================================================================
  
  inline __umesh_both__
  void computeUniqueVertexOrder(Facet &facet)
  {
    int4 idx = facet.vertexIdx;
    if (idx.w < 0) {
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
    facet.vertexIdx = idx;
  }

#ifdef __CUDACC__
  __global__
  void computeUniqueVertexOrderLaunch(Facet *facets, size_t numFacets)
  {
    size_t jobIdx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (jobIdx >= numFacets)
      return;
    computeUniqueVertexOrder(facets[jobIdx]);
  }

  void computeUniqueVertexOrder(Facet *facet, size_t numFacets)
  {
    size_t blockSize = 128;
    size_t numBlocks = divRoundUp(numFacets,blockSize);
    computeUniqueVertexOrderLaunch<<<(int)numBlocks,(int)blockSize>>>
      (facet,numFacets);
  }
#else
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
#endif

  // ==================================================================
  // init faces
  // ==================================================================
  
  inline __umesh_both__
  void writeTetFacets(Facet *facets,
                      size_t tetIdx,
                      InputMesh mesh
                      )
  {
    for (int i=0;i<4;i++) facets[i].prim.primType = UMesh::TET;
    for (int i=0;i<4;i++) facets[i].prim.facetIdx = i;
    for (int i=0;i<4;i++) facets[i].prim.primIdx  = tetIdx;
    for (int i=0;i<4;i++) facets[i].orientation   = 0;
    
    UMesh::Tet tet = mesh.tets[tetIdx];
    facets[0].vertexIdx = { tet.y,tet.w,tet.z,-1 };
    facets[1].vertexIdx = { tet.x,tet.z,tet.w,-1 };
    facets[2].vertexIdx = { tet.x,tet.w,tet.y,-1 };
    facets[3].vertexIdx = { tet.x,tet.y,tet.z,-1 };
  }
  
  inline __umesh_both__
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
  
  inline __umesh_both__
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
  
  inline __umesh_both__
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
  
  inline __umesh_both__
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

#ifdef __CUDACC__
  __global__
  void writeFacetsLaunch(Facet *facets,
                         InputMesh mesh)
  {
    size_t jobIdx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    writeFacets(facets,jobIdx,mesh);
  }
  void writeFacets(Facet *facets,
                   const InputMesh &mesh)
  {
    size_t numPrims
      = mesh.numTets
      + mesh.numPyrs
      + mesh.numWedges
      + mesh.numHexes;
    size_t blockSize = 128;
    size_t numBlocks = divRoundUp(numPrims,blockSize);
    writeFacetsLaunch<<<(int)numBlocks,(int)blockSize>>>(facets,mesh);
  }
#else
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
#endif

  // ==================================================================
  // sort facet array
  // ==================================================================
#ifdef __CUDACC__
  void sortFacets(Facet *facets, size_t numFacets)
  {
    thrust::sort(thrust::device,
                 facets,facets+numFacets,FacetComparator());
  }
#else
  void sortFacets(Facet *facets, size_t numFacets)
  {
# ifdef UMESH_HAVE_TBB
    tbb::parallel_sort(facets,facets+numFacets,FacetComparator());
# else
    std::sort(facets,facets+numFacets,FacetComparator());
# endif
    // for (int i=0;i<100;i++)
    //   std::cout << "facet " << i << " " << facets[i].vertexIdx << std::endl;
  }
#endif
  
  // ==================================================================
  // set up / upload input mesh data
  // ==================================================================
#ifdef __CUDACC__
  template<typename T>
  inline void upload(T *&ptr, size_t &count,
                     const std::vector<T> &vec)
  {
    count = vec.size();
    if (count == 0)
      ptr = 0;
    else {
      CUDA_CALL(MallocManaged((void**)&ptr,count*sizeof(T)));
      CUDA_CALL(Memcpy(ptr,vec.data(),count*sizeof(T),cudaMemcpyDefault));
    }
  }
  inline void freeInput(InputMesh &mesh)
  {
    CUDA_CALL(Free(mesh.tets));
    CUDA_CALL(Free(mesh.pyrs));
    CUDA_CALL(Free(mesh.wedges));
    CUDA_CALL(Free(mesh.hexes));
  }
  Facet *allocateFacets(size_t numFacets)
  {
    Facet *facets;
    CUDA_CALL(MallocManaged((void**)&facets,numFacets*sizeof(Facet)));
    return facets;
  }
  void freeFacets(Facet *facets)
  { CUDA_CALL(Free(facets)); }
#else
  template<typename T>
  inline void upload(T *&ptr, size_t &count,
                     const std::vector<T> &vec)
  {
    ptr = (T*)vec.data();
    count = vec.size();
  }
  inline void freeInput(InputMesh &mesh)
  {}
  Facet *allocateFacets(size_t numFacets) { return new Facet[numFacets]; }
  void freeFacets(Facet *facets)
  { delete[] facets; }
#endif

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
  inline __umesh_both__
  void facesWriteFacesKernel(SharedFace *faces,
                             const Facet *facets,
                             const uint64_t *faceIndices,
                             size_t facetIdx)
  {
    const Facet facet = facets[facetIdx];
    size_t faceIdx = faceIndices[facetIdx]-1;
    SharedFace &face = faces[faceIdx];
    auto &side = facet.orientation ? face.onFront : face.onBack;
    face.vertexIdx = facet.vertexIdx;
    side = facet.prim;
  }
  
#ifdef __CUDACC__
  __global__ void facesWriteFacesLaunch(SharedFace *faces,
                                        const Facet *facets,
                                        const uint64_t *faceIndices,
                                        size_t numFacets)
  {
    size_t jobIdx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (jobIdx >= numFacets) return;
    facesWriteFacesKernel(faces,facets,faceIndices,jobIdx);
  }
  
  void facetsWriteFaces(SharedFace *faces,
                        const Facet *facets,
                        const uint64_t *faceIndices,
                        size_t numFacets)
  {
    size_t blockSize = 128;
    size_t numBlocks = divRoundUp(numFacets,blockSize);
    facesWriteFacesLaunch<<<(int)numBlocks,(int)blockSize>>>
      (faces,facets,faceIndices,numFacets);
  }
#else
  void facetsWriteFaces(SharedFace *faces,
                        const Facet *facets,
                        const uint64_t *faceIndices,
                        size_t numFacets)
  {
    parallel_for_blocked
      (0,numFacets,1024,
       [&](size_t begin, size_t end) {
         for (size_t i=begin;i<end;i++)
           facesWriteFacesKernel(faces,facets,faceIndices,i);
       });
  }
#endif
  
  // ==================================================================
  // manage mem for faces
  // ==================================================================
#ifdef __CUDACC__
  SharedFace *allocateFaces(std::vector<SharedFace> &result,
                            size_t numFaces)
  {
    SharedFace *ptr;
    CUDA_CALL(MallocManaged((void**)&ptr,numFaces*sizeof(*ptr)));
    return ptr;
  }
  
  void finishFaces(std::vector<SharedFace> &result,
                   SharedFace *faces,
                   size_t numFaces)
  {
    result.resize(numFaces);
    CUDA_CALL(Memcpy(result.data(),faces,
                     numFaces*sizeof(*faces),cudaMemcpyDefault));
    CUDA_CALL(Free(faces));
  }
#else
  SharedFace *allocateFaces(std::vector<SharedFace> &result,
                            size_t numFaces)
  {
    result.resize(numFaces);
    return result.data();
  }
  
  void finishFaces(std::vector<SharedFace> &result,
                   SharedFace *faces,
                   size_t numFaces)
  {
#if 0
    /* validate - make sure to have this off in releases */
    std::set<vec4i> knownFaces;
    for (int i=0;i<numFaces;i++) {
      vec4i idx = faces[i].vertexIdx;
      std::sort(&idx.x,&idx.x+4);
      if (knownFaces.find(idx) != knownFaces.end())
        std::cout << "validation failed: given face already exists : " << faces[i].vertexIdx << std::endl;
      knownFaces.insert(idx);
    }
    std::cout << "done validation, found " << knownFaces.size() << " unique faces" << std::endl;
#endif
    /* nothing to do */
  }
#endif

  // ==================================================================
  // manage mem for face indices
  // ==================================================================

#ifdef __CUDACC__
  void freeIndices(uint64_t *faceIndices)
  {
    CUDA_CALL(Free(faceIndices));
  }
  uint64_t *allocateIndices(size_t numFacets)
  {
    uint64_t *ptr;
    CUDA_CALL(MallocManaged((void**)&ptr,numFacets*sizeof(*ptr)));
    return ptr;
  }
#else
  void freeIndices(uint64_t *faceIndices)
  {
    delete[] faceIndices;
  }
  uint64_t *allocateIndices(size_t numFacets)
  {
    return new uint64_t[numFacets];
  }
#endif
  
  // ==================================================================
  // compute face indices from (sorted) facet array
  // ==================================================================

  inline __umesh_both__
  void initFaceIndexKernel(uint64_t *faceIndices,
                           Facet *facets,
                           size_t facetIdx)
  {
    faceIndices[facetIdx]
      =  facetIdx == 0
      || facets[facetIdx-1].vertexIdx.x != facets[facetIdx].vertexIdx.x
      || facets[facetIdx-1].vertexIdx.y != facets[facetIdx].vertexIdx.y
      || facets[facetIdx-1].vertexIdx.z != facets[facetIdx].vertexIdx.z
      || facets[facetIdx-1].vertexIdx.w != facets[facetIdx].vertexIdx.w;
  }

#ifdef __CUDACC__
  void clearFaces(SharedFace *faces, size_t numFaces)
  {
    CUDA_CALL(Memset(faces,-1,numFaces*sizeof(SharedFace)));
  }

  __global__
  void initFaceIndicesLaunch(uint64_t *faceIndices,
                             Facet *facets,
                             size_t numFacets)
  {
    size_t facetIdx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (facetIdx >= numFacets) return;
    initFaceIndexKernel(faceIndices,facets,facetIdx);
  }
  
  void initFaceIndices(uint64_t *faceIndices,
                       Facet *facets,
                       size_t numFacets)
  {
    size_t blockSize = 128;
    size_t numBlocks = divRoundUp(numFacets+1,blockSize);
    initFaceIndicesLaunch<<<(int)numBlocks,(int)blockSize>>>
      (faceIndices,facets,numFacets);
  }
  void prefixSum(uint64_t *faceIndices,
                 size_t numFacets)
  {
    thrust::exclusive_scan(thrust::device,
                           faceIndices,faceIndices+numFacets,
                           faceIndices);
  }
  void postfixSum(uint64_t *faceIndices,
                 size_t numFacets)
  {
    thrust::inclusive_scan(thrust::device,
                           faceIndices,faceIndices+numFacets,
                           faceIndices);
  }
#else
  void clearFaces(SharedFace *faces, size_t numFaces)
  {
    PrimFacetRef clearPrim = { 0,0,-1 };
    for (size_t i=0;i<numFaces;i++) {
      faces[i].onFront = faces[i].onBack = clearPrim;
    }
  }
  void initFaceIndices(uint64_t *faceIndices,
                       Facet *facets,
                       size_t numFacets)
  {
    for (size_t i=0;i<numFacets;i++) {
      initFaceIndexKernel(faceIndices,facets,i);
    }
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
  /*! not parallelized... this will likely be mem bound, anyway */
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
#endif
  


  // ==================================================================
  // aaaand ... wrap it all together
  // ==================================================================

  std::vector<SharedFace> computeFaces(UMesh::SP input)
  {
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
    Facet *facets = allocateFacets(numFacets);
    writeFacets(facets,mesh);
    computeUniqueVertexOrder(facets,numFacets);
    
    // -------------------------------------------------------
    sortFacets(facets,numFacets);
    uint64_t *faceIndices = allocateIndices(numFacets);
    initFaceIndices(faceIndices,facets,numFacets);
    postfixSum(faceIndices,numFacets);
    // prefixSum(faceIndices,numFacets);
    // -------------------------------------------------------
    size_t numFaces = faceIndices[numFacets-1]+1;
    std::vector<SharedFace> result;
    SharedFace *faces = allocateFaces(result,numFaces);
    clearFaces(faces,numFaces);
    
    // -------------------------------------------------------
    facetsWriteFaces(faces,facets,faceIndices,numFacets);
#ifdef __CUDACC__
    CUDA_SYNC_CHECK();
#endif
    std::chrono::steady_clock::time_point
      end_exc = std::chrono::steady_clock::now();
    
    finishFaces(result,faces,numFaces);
    freeIndices(faceIndices);

    std::chrono::steady_clock::time_point
      end_inc = std::chrono::steady_clock::now();
    std::cout << "done computing faces, including upload/download "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_inc - begin_inc).count()/1024.f << " secs, vs excluding "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_exc - begin_exc).count()/1024.f  << std::endl;
    return result;
  }
  
  extern "C" int main(int ac, char **av)
  {
      try {
          std::string inFileName, outFileName;
          for (int i = 1; i < ac; i++) {
              const std::string arg = av[i];
              if (arg == "-o")
                  outFileName = av[++i];
              else if (arg[0] == '-')
                  usage("unknown cmdline argument " + arg);
              else
                  inFileName = arg;
          }
          if (inFileName == "")
              throw std::runtime_error("no test file specified");
          UMesh::SP input = UMesh::loadFrom(inFileName);
          std::vector<SharedFace> result
              = computeFaces(input);
          std::cout << "done computing shared faces, found " << result.size() << " faces for mesh of " << input->toString() << std::endl;
      }
      catch (std::exception e) {
          std::cerr << "fatal error " << e.what() << std::endl;
      }
    return 0;
  } 
  
}
