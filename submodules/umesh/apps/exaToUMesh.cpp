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

#include "umesh/UMesh.h"
#include "umesh/io/IO.h"
#include "umesh/check.h"
// #include "tetty/UMesh.h"
#include <set>
#include <map>
#include <fstream>
#include <atomic>
#include <array>

#define DEBUG 0

#if DEBUG
# define DBG(a) a
#else
# define DBG(a) /**/
#endif

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
  std::atomic<uint64_t> numTets;
  std::atomic<uint64_t> numWedges;
  std::atomic<uint64_t> numWedgesPerfect;
  std::atomic<uint64_t> numWedgesTwisted;

  std::atomic<uint64_t> numPyramids;
  std::atomic<uint64_t> numPyramidsPerfect;
  std::atomic<uint64_t> numPyramidsTwisted;

  std::atomic<uint64_t> numHexes;
  std::atomic<uint64_t> numHexesPerfect;
  std::atomic<uint64_t> numHexesTwisted;

  /*! if set, we will ONLY emit boundary elements */
  bool boundaryOnly = false;
}
inline bool operator<(const umesh::Tet &a, const umesh::Tet &b)
{
  const uint64_t *pa = (const uint64_t *)&a;
  const uint64_t *pb = (const uint64_t *)&b;
  const bool res =  (pa[0] < pb[0]) || ((pa[0] == pb[0]) && (pa[1] < pb[1]));
  return res;
}
// namespace std {
//   inline bool operator<(const umesh::vec4f a, const umesh::vec4f b)
//   {
//     for (int i=0;i<4;i++) {
//       if (a[i] < b[i]) return true;
//       if (a[i] > b[i]) return false;
//     }
//     return false;
//   }
// }


namespace umesh {

  std::map<vec3i,std::vector<std::pair<vec4i,int>>> alreadyGeneratedFaces;
  
  struct Exa {
    struct LogicalCell {
      inline box3f bounds() const
      { return box3f(vec3f(pos),vec3f(pos+vec3i(1<<level))); }
      inline LogicalCell neighbor(const vec3i &delta) const
      { return { pos+delta*(1<<level),level }; }
      
      inline vec3f center() const { return vec3f(pos) + vec3f(0.5f*(1<<level)); }
      vec3i pos;
      int   level;
    };
  
    struct Cell : public LogicalCell {
      float scalar;
    };
  
    void add(Cell cell)
    {
      // cells[cell] = (int)cellList.size();
      cellList.push_back(cell);
      // if (cells.size() != cellList.size())
      //   throw std::runtime_error("bug in add(cell)");
      minLevel = min(minLevel,cell.level);
      maxLevel = max(maxLevel,cell.level);
      bounds.extend(cell.bounds());
    }

    size_t size() const { return cellList.size(); }

    box3f bounds;
    int minLevel=100, maxLevel=0;
    // stores cell and ID
    std::vector<Cell>  cellList;
    // std::map<Cell,size_t> cells;
    
    bool find(int &cellID, const vec3f &pos) const;
  };

  inline bool operator<(const Exa::LogicalCell &a, const Exa::LogicalCell &b)
  {
    const uint64_t *pa = (uint64_t *)&a;
    const uint64_t *pb = (uint64_t *)&b;
    return
      (pa[0] < pb[0])
      || (pa[0] == pb[0] && pa[1] < pb[1]);
  }

  inline bool operator==(const Exa::LogicalCell &a, const Exa::LogicalCell &b)
  {
    const uint64_t *pa = (uint64_t *)&a;
    const uint64_t *pb = (uint64_t *)&b;
    return pa[0] == pb[0] && pa[1] == pb[1];
  }

  inline bool operator<(const Exa::Cell &a, const Exa::Cell &b)
  {
    const Exa::LogicalCell &la = a;
    const Exa::LogicalCell &lb = b;
  
    return (la < lb) || ((la==lb) && (a.scalar < b.scalar));
  }

  inline bool operator==(const Exa::Cell &a, const Exa::Cell &b)
  {
    return ((const Exa::LogicalCell &)a == (const Exa::LogicalCell &)b) && (a.scalar == b.scalar);
  }


  inline bool operator!=(const Exa::Cell &a, const Exa::Cell &b)
  {
    return !(a == b);
  }

  using namespace std;
  
  std::ostream &operator<<(std::ostream &out, const Exa::LogicalCell &cell)
  {
    out << "[" << cell.pos << ","<<cell.level <<"]";
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const Exa::Cell &cell)
  {
    out << "[" << cell.pos << ","<<cell.level <<";" << cell.scalar << "]";
    return out;
  }


  int lowerOnLevel(float f, int level)
  {
    f = floorf(f/(1<<level));
    return f*(1<<level);
  }

  // return vector-index of given cell, if exists, or -1
  bool Exa::find(int &result, const vec3f &where) const
  {
    // std::cout << "=======================================================" << std::endl;
    // // dbg = true;
    // PING;
    DBG(PING; PRINT(where));
    for (int level=minLevel;level<=maxLevel;level++) {
      // std::cout << "-------------------------------------------------------" << std::endl;
      int width = 1<<level;
      Cell query;
      query.pos.x = lowerOnLevel(where.x,level); //int(floorf(where.x)) & ~(width-1);
      query.pos.y = lowerOnLevel(where.y,level); //int(floorf(where.y)) & ~(width-1);
      query.pos.z = lowerOnLevel(where.z,level); //int(floorf(where.z)) & ~(width-1);
      query.level = level;
      auto it = std::lower_bound(cellList.begin(),cellList.end(),query,
                                 [&](const Cell &a, const Cell &b)->bool{return (const Exa::LogicalCell&)a < (const Exa::LogicalCell&)b;});
      DBG(PRINT(level));
      DBG(PRINT(query));
      DBG(PRINT(*it));
      if (it != cellList.end() && (const LogicalCell&)*it == (const LogicalCell&)query) {
        result = it-cellList.begin();
        return true;
      }
    }
    result   = -1;
    return false;
  }


  // ##################################################################
  // managing output vertex and scalar generation
  // ##################################################################

  std::map<vec3f,int> vertexIndex;
  std::mutex vertexMutex;

  std::shared_ptr<UMesh> output;
  std::mutex outputMutex;

  int findOrEmitVertex(const vec4f &v)
  {
    std::lock_guard<std::mutex> lock(vertexMutex);

    auto it = vertexIndex.find((const vec3f&)v);
    if (it != vertexIndex.end()) return it->second;
  
    size_t newID = output->vertices.size();
    output->vertices.push_back(vec3f{v.x, v.y, v.z});
    output->perVertex->values.push_back(v.w);
  
    if (newID >= 0x7fffffffull) {
      PING;
      throw std::runtime_error("vertex index overflow ...");
    }
  
    vertexIndex[(const vec3f&)v] = (int)newID;
    return newID;
  }











  /*! tests if the given four vertices are a plar dual-grid face - note
    this will ONLY wok for (possibly degen) dual cells, it will _NOT_
    do a general planarity test (eg, it would not detect rotations of
    the vertices) */
  template<int U, int V>
  inline bool isPlanarQuadFaceT(const vec4f v0,
                                const vec4f v1,
                                const vec4f v2,
                                const vec4f v3)
  {
    const vec2f v00 = vec2f(v0[U],v0[V]);
    const vec2f v01 = vec2f(v1[U],v1[V]);
    const vec2f v10 = vec2f(v3[U],v3[V]);
    const vec2f v11 = vec2f(v2[U],v2[V]);
    return
      (v00 == v01 && v10 == v11) ||
      (v00 == v10 && v01 == v11);
  }

  /*! tests if the given four vertices are a plar dual-grid face - note
    this will ONLY wok for (possibly degen) dual cells, it will _NOT_
    do a general planarity test (eg, it would not detect rotations of
    the vertices) */
  bool isPlanarQuadFace(const vec4f base00,
                        const vec4f base01,
                        const vec4f base11,
                        const vec4f base10)
  {
    return
      isPlanarQuadFaceT<0,1>(base00,base01,base11,base10) ||
      isPlanarQuadFaceT<0,2>(base00,base01,base11,base10) ||
      isPlanarQuadFaceT<1,2>(base00,base01,base11,base10) ||
      // mirror
      isPlanarQuadFaceT<1,0>(base00,base01,base11,base10) ||
      isPlanarQuadFaceT<2,0>(base00,base01,base11,base10) ||
      isPlanarQuadFaceT<2,1>(base00,base01,base11,base10);
  }




  // ##################################################################
  // actual 'emit' functions - these *will* write the specified prim w/o
  // any other tests
  // ##################################################################

  void printCounts()
  {
    std::cout << "generated "
              << prettyNumber(numTets) << " tets, "
    
              << prettyNumber(numPyramids) << " pyramids ("
              << prettyNumber(numPyramidsPerfect) << " perfect, " 
              << prettyNumber(numPyramidsTwisted) << " twisted), " 

              << prettyNumber(numWedges) << " wedges ("
              << prettyNumber(numWedgesPerfect) << " perfect, " 
              << prettyNumber(numWedgesTwisted) << " twisted), " 

              << prettyNumber(numHexes) << " hexes ("
              << prettyNumber(numHexesPerfect) << " perfect, " 
              << prettyNumber(numHexesTwisted) << " twisted)." 
              << std::endl;
  }


  void sanityCheckFace(vec3i face, const vec4i &tet, int pyrTop)
  {
#if 1
    return;
#else
    static std::mutex mutex;
    std::sort(&face.x,&face.x+3);

    std::lock_guard<std::mutex> lock(mutex);
    alreadyGeneratedFaces[face].push_back({tet,pyrTop});
    if (alreadyGeneratedFaces[face].size() > 2) {
      PRINT(face);
      for (auto prim : alreadyGeneratedFaces[face]) {
        PRINT(prim.first);
        PRINT(prim.second);
      }
      throw std::runtime_error("face generated more than once!");
      exit(1);
    }
#endif
  }
  void sanityCheckTet(const vec4i &tet)
  {
    assert(tet.x != tet.y);
    assert(tet.x != tet.z);
    assert(tet.x != tet.w);

    assert(tet.y != tet.z);
    assert(tet.y != tet.w);

    assert(tet.z != tet.w);

    sanityCheckFace({tet.x,tet.y,tet.z},tet,-1);
  }
  
  void emitTet(const std::array<vec4f,4> &vertices)
  {
    const vec4f &A = vertices[0];    
    const vec4f &B = vertices[1];    
    const vec4f &C = vertices[2];    
    const vec4f &D = vertices[3];
    
    const vec4i tet(findOrEmitVertex(A),
                    findOrEmitVertex(B),
                    findOrEmitVertex(C),
                    findOrEmitVertex(D));
  
    sanityCheckTet(tet);
    
    std::lock_guard<std::mutex> lock(outputMutex);
    output->tets.push_back({(int)tet.x, (int)tet.y, (int)tet.z, (int)tet.w});
  
    static int nextPing = 1;
    if (numTets++ >= nextPing) {
      nextPing*=2;
      printCounts();
    }
  };

  // ##################################################################
  void emitPyramid(const std::array<vec4f,4> &base,
                   const vec4f &top)
  {
    UMesh::Pyr pyr;
    pyr[4]    = findOrEmitVertex(top);
    pyr[0] = findOrEmitVertex(base[0]);
    pyr[1] = findOrEmitVertex(base[1]);
    pyr[2] = findOrEmitVertex(base[2]);
    pyr[3] = findOrEmitVertex(base[3]);

    if (isPlanarQuadFace(base[0],base[1],base[2],base[3]))
      numPyramidsPerfect++;
    else
      numPyramidsTwisted++;

    sanityCheckFace({pyr[0],pyr[1],pyr[4]},(const vec4i&)pyr, pyr[4]);
    sanityCheckFace({pyr[1],pyr[2],pyr[4]},(const vec4i&)pyr, pyr[4]);
    sanityCheckFace({pyr[2],pyr[3],pyr[4]},(const vec4i&)pyr, pyr[4]);
    sanityCheckFace({pyr[3],pyr[0],pyr[4]},(const vec4i&)pyr, pyr[4]);
    
    std::lock_guard<std::mutex> lock(outputMutex);
    output->pyrs.push_back(pyr);

    static int nextPing = 1;
    if (numPyramids++ >= nextPing) {
      nextPing*=2;
      printCounts();
    }
  }

  void emitWedge(const std::array<vec4f,3> &front,
                 const std::array<vec4f,3> &back)
  {
    UMesh::Wedge wedge;
    wedge[0] = findOrEmitVertex(front[0]);
    wedge[1] = findOrEmitVertex(front[1]);
    wedge[2] = findOrEmitVertex(front[2]);
    wedge[3] = findOrEmitVertex(back[0]);
    wedge[4] = findOrEmitVertex(back[1]);
    wedge[5] = findOrEmitVertex(back[2]);

    if (isPlanarQuadFace(front[0],front[1],back[0],back[1]) &&
        isPlanarQuadFace(front[0],front[2],back[0],back[2]) &&
        isPlanarQuadFace(front[1],front[2],back[1],back[2]))
      numWedgesPerfect++;
    else
      numWedgesTwisted++;
    
    std::lock_guard<std::mutex> lock(outputMutex);
    output->wedges.push_back(wedge);
  
    static int nextPing = 1;
    if (numWedges++ >= nextPing) {
      nextPing*=2;
      printCounts();
    }
  }





  void emitHex(const std::array<vec4f,8> corner, bool perfect)
  {
    UMesh::Hex hex;
    // vtk order:
    hex[0] = findOrEmitVertex(corner[0]);
    hex[1] = findOrEmitVertex(corner[1]);
    hex[2] = findOrEmitVertex(corner[2]);
    hex[3] = findOrEmitVertex(corner[3]);
  
    hex[4] = findOrEmitVertex(corner[4]);
    hex[5] = findOrEmitVertex(corner[5]);
    hex[6] = findOrEmitVertex(corner[6]);
    hex[7] = findOrEmitVertex(corner[7]);
  
  
    std::lock_guard<std::mutex> lock(outputMutex);
    output->hexes.push_back(hex);

    if (perfect)
      numHexesPerfect++;
    else
      numHexesTwisted++;
  
    static int nextPing = 1;
    if (numHexes++ >= nextPing) {
      nextPing*=2;
      printCounts();
    }
  }




  // /*! check if the given four vertices form a valid tet, and if so, emit
  //   (else just drop it) */
  // void tryTet(const vec4f &A,
  //             const vec4f &B,
  //             const vec4f &C,
  //             const vec4f &D)
  // {
  //   if (A == B) return;
  //   if (A == C) return;
  //   if (A == D) return;
  //   if (B == C) return;
  //   if (B == D) return;
  //   if (C == D) return;
  //   emitTet(A,B,C,D);
  // }


  /*! if this gets called we know that one side of a general dual cell
    has collapsed into a single vertex (the 'top' here), but the
    other four could still have duplicates .... we further do know
    that the base face has NOT collapsed completely (else we'd have
    had more than 5 duplicates, which gets tested first) */
  void tryPyramid(const std::array<vec4f,4> &base,
                  const vec4f &top,
                  int numUniqueVertices)
  {
    if (numUniqueVertices == 5) {
      // MUST be a pyramid
      emitPyramid(base,top);
      return;
    }

    if (numUniqueVertices == 4) {
      // check if any of the EDGES of the base collapsed, then it's a tet.
      if (base[0]==base[1]) {
        emitTet({base[1],base[2],base[3],top});
        return;
      }
      if (base[1]==base[2]) {
        emitTet({base[2],base[3],base[0],top});
        return;
      }

      if (base[2]==base[3]) {
        emitTet({base[3],base[0],base[1],top});
        return;
      }
      
      if (base[3]==base[0]) {
        emitTet({base[0],base[1],base[2],top});
        return;
      }
      
      // if not, it must be to opposite vertice in the bottom face
      // that collapsed, then this is totally degen.

      if (base[0] == base[2])
        // degen, ignore
        return;

      if (base[1] == base[3])
        // degen, ignore
        return;
      
      throw std::runtime_error("this case should not happen!?");
    }
    
    throw std::runtime_error("this cannot happen!?");
    
    // PING; PRINT(numUniqueVertices);
    
    // // apparentlythis does happen:
    // if ((base00 == base11) ||
    //     (base10 == base01)) {
    //   PRINT(top);
    //   PRINT(base00);
    //   PRINT(base01);
    //   PRINT(base10);
    //   PRINT(base11);
    //   throw std::runtime_error("really!?");
    //   return;
    // }
    // xxx
    // if (base00 == base01) {
    //   tryTet(top,base00,base10,base11);
    //   return;
    // }
    // if (base00 == base10) {
    //   tryTet(top,base00,base01,base11);
    //   return;
    // }
    // if (base11 == base10) {
    //   tryTet(top,base11,base01,base00);
    //   return;
    // }
    // if (base11 == base01) {
    //   tryTet(top,base11,base10,base00);
    //   return;
    // }

    // emitPyramid(top,base00,base01,base10,base11);
  }

  // void tryWedge(const vec4f &top0,
  //               const vec4f &top1,
  //               const vec4f &base00,
  //               const vec4f &base01,
  //               const vec4f &base10,
  //               const vec4f &base11,
  //               int numUniqueVertices)
  // {
  //   if (numUniqueVertices == 2) {
  //     // std::cout << "REAL wedge" << std::endl;
  //     emitWedge(top0,top1,base00,base01,base10,base11);
  //     return;
  //   }

  //   // OK, if there's MORE than two duplicates then the only way the
  //   // bottom face could be a true bilinear patch is if the top was a
  //   // single point - which it isn't, or it'd have been catched in the
  //   // pyramid cases. As such, the bottom is either a triangle, or
  //   // totally degenerate, but etiher way we can tessellate it into tets
  //   // std::cout << "splittin wedge into pyramid plus tet..." << std::endl;
  //   tryTet(top0,top1,base10,base11);
  //   tryPyramid(top0,base00,base01,base10,base11);
  // }


  /*! if this gets called we know that at least one face has collapsed
    to an edge, but that NOT an entire face has collapsed (the
    latter was tested before testing for edges). so we know that the
    front[2] and back[2] must be different (else that face would
    have collapsed)...BUT we could still have other collapses going
    on on the 'base' spanned by front[0],front[1],back[0],back[1]
    (vertices 0,1,3,4 in vtk corder) */
  void tryWedge(const std::array<vec4f,8> &corner,
                const vec3i &frontIdx,
                const vec3i &backIdx,
                int numUniqueVertices)
  {
    // if (numUniqueVertices == 6) {
    // have 6 vertices, and already know the two that collapsed, so
    // MUST be a wedge - possibly curved faces, but that's a
    // differnt story.
    emitWedge
      ({corner[frontIdx.x],
        corner[frontIdx.y],
        corner[frontIdx.z]},
        {corner[backIdx.x],
         corner[backIdx.y],
         corner[backIdx.z]});
    return;
    // }

    // if (numUniqueVertices == 5) {
    //   // five unique vertices - we know the top is two, so the base
    //   // must be 3, and form a triangle
    //   PING; PRINT(numUniqueVertices);

    //   std::ofstream dbg("dbg.obj");
    //   for (auto v : corner)
    //     dbg << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    //   dbg << "f 1 2 6 5" << std::endl;
    //   dbg << "f 2 3 7 8" << std::endl;
    //   dbg << "f 1 4 5 8" << std::endl;
    //   dbg << "f 4 3 7 8" << std::endl;

    //   dbg << "f 1 2 3 4" << std::endl;
    //   dbg << "f 5 6 7 8" << std::endl;
      
    //   throw std::runtime_error("not implemented");
    //   return;
    // }
    
    // PING; PRINT(numUniqueVertices);
    // throw std::runtime_error("not implemented");
  }







  // ##################################################################
  // logic of processing a dual mesh cell
  // ##################################################################

  bool allSame(const vec4f &a, const vec4f &b, const vec4f &c, const vec4f &d)
  {
    return (a==b) && (a==c) && (a==d);
  }

  bool same(const vec4f &a, const vec4f &b)
  {
    return (a==b);
  }

  // void splitHex(const Exa &exa, const vec4f vertex[2][2][2])
  // {
  //   const vec4f &v000 = vertex[0][0][0];
  //   const vec4f &v001 = vertex[0][0][1];
  //   const vec4f &v010 = vertex[0][1][0];
  //   const vec4f &v011 = vertex[0][1][1];
  //   const vec4f &v100 = vertex[1][0][0];
  //   const vec4f &v101 = vertex[1][0][1];
  //   const vec4f &v110 = vertex[1][1][0];
  //   const vec4f &v111 = vertex[1][1][1];
  
  //   if (allSame(v000,v001,v010,v011)) {
  //     tryPyramid(v000, v100,v101,v110,v111);
  //     return;
  //   }
  //   if (allSame(v100,v101,v110,v111)) {
  //     tryPyramid(v100, v000,v001,v010,v011);
  //     return;
  //   }
  
  //   if (allSame(v000,v001,v100,v101)) {
  //     tryPyramid(v000, v010,v011,v110,v111);
  //     return;
  //   }
  //   if (allSame(v010,v011,v110,v111)) {
  //     tryPyramid(v010, v000,v001,v100,v101);
  //     return;
  //   }
  
  //   if (allSame(v000,v010,v100,v110)) {
  //     tryPyramid(v000, v001,v011,v101,v111);
  //     return;
  //   }
  //   if (allSame(v001,v011,v101,v111)) {
  //     tryPyramid(v001, v000,v010,v100,v110);
  //     return;
  //   }

  //   vec4f center(0.f);
  //   for (int iz=0;iz<2;iz++)
  //     for (int iy=0;iy<2;iy++)
  //       for (int ix=0;ix<2;ix++) {
  //         center = center + vertex[iz][iy][ix];
  //       }
  //   center = center * vec4f(1.f / 8.f);

  //   tryPyramid(center, v000, v001, v010, v011);
  //   tryPyramid(center, v100, v101, v110, v111);

  //   tryPyramid(center, v000, v001, v100, v101);
  //   tryPyramid(center, v010, v011, v110, v111);
  
  //   tryPyramid(center, v000, v010, v100, v110);
  //   tryPyramid(center, v001, v011, v101, v111);
  // }



  // ##################################################################
  // code that actually generates the (possibly-degenerate) dual cells
  // ##################################################################
  void doCell(const Exa &exa, const Exa::Cell &cell)
  {
    int selfID;
    exa.find(selfID,cell.center());
    if (selfID < 0 || exa.cellList[selfID] != cell)
      throw std::runtime_error("bug in exa::find()");

    const int cellWidth = (1<<cell.level);
    bool dbg = false;
    
    // if (cell.center() == vec3f(1,1,9)) dbg = true;
    
    for (int dz=-1;dz<=1;dz+=2)
      for (int dy=-1;dy<=1;dy+=2)
        for (int dx=-1;dx<=1;dx+=2) {
          if (dbg)
            std::cout << "--------------------------------------------" << std::endl;
          int corner[2][2][2];
          int minLevel = 1000;
          int maxLevel = -1;
          int numFound = 0;
          for (int iz=0;iz<2;iz++)
            for (int iy=0;iy<2;iy++)
              for (int ix=0;ix<2;ix++) {
                // PRINT(vec3i(ix,iy,iz));
                const vec3f cornerCenter = cell.neighbor(vec3i(dx*ix,dy*iy,dz*iz)).center();
                // if (cornerCenter == vec3f(1,1,9)) dbg = true;
                
                // PRINT(cornerCenter);
                if (!exa.find(corner[iz][iy][ix],cornerCenter))
                  // corner does not exist, this is not a dual cell
                  continue;
              
                minLevel = min(minLevel,exa.cellList[corner[iz][iy][ix]].level);
                maxLevel = max(maxLevel,exa.cellList[corner[iz][iy][ix]].level);
                ++numFound;
              }

          if (dbg) {
            PING;
            PRINT(cell);
            PRINT(vec3i(dx,dy,dz));
            for (int iz=0;iz<2;iz++)
              for (int iy=0;iy<2;iy++)
                for (int ix=0;ix<2;ix++)
                  PRINT(corner[iz][iy][ix]);
            PRINT(numFound);
            PRINT(cell);
            PRINT(minLevel);
          }
          
          if (numFound < 8)
            continue;
          
          if (minLevel < cell.level)
            // somebody else will generate this same cell from a finer
            // level...
            continue;

          
          Exa::Cell minCell = cell;
          for (int iz=0;iz<2;iz++)
            for (int iy=0;iy<2;iy++)
              for (int ix=0;ix<2;ix++) {
                int cID = corner[iz][iy][ix];
                if (cID < 0) continue;
                Exa::Cell cc = exa.cellList[cID];
                if (cc.level == cell.level && cc < minCell)
                  minCell = cc;
              }

          
          if (minCell != cell)
            // some other cell will generate this
            continue;

          vec4f vertex[2][2][2];
          for (int iz=0;iz<2;iz++)
            for (int iy=0;iy<2;iy++)
              for (int ix=0;ix<2;ix++) {
                const Exa::Cell &c = exa.cellList[corner[iz][iy][ix]];
                if (dbg) PRINT(c);
                vertex[iz][iy][ix] = vec4f(c.center(),c.scalar);
              }


          // const vec4f &v000 = vertex[0][0][0];
          // const vec4f &v001 = vertex[0][0][1];
          // const vec4f &v010 = vertex[0][1][0];
          // const vec4f &v011 = vertex[0][1][1];
          // const vec4f &v100 = vertex[1][0][0];
          // const vec4f &v101 = vertex[1][0][1];
          // const vec4f &v110 = vertex[1][1][0];
          // const vec4f &v111 = vertex[1][1][1];

          // VTK order
          std::array<vec4f,8> v;
          if ((dx<0) ^ (dy<0) ^ (dz<0)) {
            // hex is mirrored an un-even time, so has negative volume... swap
            v[0] = vertex[1][0][0];
            v[1] = vertex[1][0][1];
            v[2] = vertex[1][1][1];
            v[3] = vertex[1][1][0];
            v[4] = vertex[0][0][0];
            v[5] = vertex[0][0][1];
            v[6] = vertex[0][1][1];
            v[7] = vertex[0][1][0];
          } else {
            v[0] = vertex[0][0][0];
            v[1] = vertex[0][0][1];
            v[2] = vertex[0][1][1];
            v[3] = vertex[0][1][0];
            v[4] = vertex[1][0][0];
            v[5] = vertex[1][0][1];
            v[6] = vertex[1][1][1];
            v[7] = vertex[1][1][0];
          }

          if (dbg) {
            PING;
            PRINT(cell);
            PRINT(cellWidth);
            PRINT(dx);
            PRINT(dy);
            PRINT(dz);
            for (auto vtx : v)
              PRINT(vtx);
          }
          
          std::set<vec4f> uniqueVertices;
          for (auto vtx : v)
            uniqueVertices.insert(vtx);
          int numUniqueVertices = uniqueVertices.size();

          const auto &v0 = v[0];
          const auto &v1 = v[1];
          const auto &v2 = v[2];
          const auto &v3 = v[3];
          const auto &v4 = v[4];
          const auto &v5 = v[5];
          const auto &v6 = v[6];
          const auto &v7 = v[7];

          // ==================================================================
          // check for regular cube
          // ==================================================================
          if (minLevel == maxLevel) {
            if (!boundaryOnly) {
              // PING; std::cout << "PERFECT HEX" << std::endl;
              emitHex(v,/*perfect:*/true);
            }
            continue;
            // return;
          }
          // ==================================================================
          // check for general hex (with possibly twisted sides)
          // ==================================================================
          // no duplicates, MUST be a general hex
          if (numUniqueVertices == 8) {
            emitHex(v,/*perfect:*/false);
            continue;
            // return;
          }

          // ==================================================================
          // check for totally degenerate
          // ==================================================================
          if (numUniqueVertices < 4) {
            // check for less than four vertices .... that cannot even
            // be a tet ... though even for exactly four it's not sure
            // it's a tet, so let's handle that int the other cases
            continue;
            // return;
          }

          // from here on, numunique = 4,5,6,or 7 are still all valid
          
          // ==================================================================
          // check whether an entire face completely collapsed - then
          // it's a pyramid (numunique==5), or a tet
          // (numunique==4). (less than pyramid or tet would mean
          // numunique<4, which has already been tested above)
          // ==================================================================
          // bottom:
          if (allSame(v0,v1,v2,v3)) {
            tryPyramid(/*facing down:*/{ v4,v7,v6,v5 }, v0, numUniqueVertices);
            continue;
            // return;
          }
          // top:
          if (allSame(v4,v5,v6,v7)) {
            tryPyramid(/* up:*/{ v0,v1,v2,v3 }, v4, numUniqueVertices);
            continue;
            // return;
          }
          // front:
          if (allSame(v0,v1,v4,v5)) {
            tryPyramid(/* face forward*/{v2,v6,v7,v3}, v0, numUniqueVertices);
            continue;
            // return;
          }
          // back:
          if (allSame(v2,v3,v6,v7)) {
            tryPyramid(/* face back*/{v0,v4,v5,v1}, v2, numUniqueVertices);
            continue;
            // return;
          }
          //left:
          if (allSame(v0,v3,v4,v7)) {
            tryPyramid(/* face right*/{v1,v5,v6,v2}, v0, numUniqueVertices);
            continue;
            // return;
          }
          //right:
          if (allSame(v1,v2,v5,v6)) {
            tryPyramid(/* face left*/{v0,v3,v7,v4}, v1, numUniqueVertices);
            continue;
            // return;
          }
        
          // ==================================================================
          // no face that completely collapsed to a single vertex -
          // now check if any one face collapsed two edges to form the
          // top of a tent - then based on what happens at the bottom
          // face it's either a wedge, a tet, or degenerate
          // ==================================================================

          // check front side:
          if (same(v0,v1) && same(v4,v5)) {
            tryWedge(v,{3,2,0},{7,6,4}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v0,v4) && same(v1,v5)) {
            tryWedge(v,{2,6,5},{3,7,4}, numUniqueVertices);
            continue;
            // return;
          }

          // check back side:
          if (same(v3,v7) && same(v2,v6)) {
            tryWedge(v,{5,1,2},{4,0,3}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v2,v3) && same(v6,v7)) {
            tryWedge(v,{1,0,3},{5,4,7}, numUniqueVertices);
            continue;
            // return;
          }

          // check top side:
          if (same(v4,v7) && same(v5,v6)) {
            tryWedge(v,{3,0,4},{2,1,6}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v4,v5) && same(v6,v7)) {
            tryWedge(v,{0,1,4},{3,2,7}, numUniqueVertices);
            continue;
            // return;
          }

          // check bottom side:
          if (same(v0,v1) && same(v3,v2)) {
            tryWedge(v,{5,4,0},{6,7,3}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v0,v3) && same(v1,v2)) {
            tryWedge(v,{4,7,3},{5,6,2}, numUniqueVertices);
            continue;
            // return;
          }

          // check left side:
          if (same(v0,v3) && same(v4,v7)) {
            tryWedge(v,{5,6,7},{1,2,3}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v0,v4) && same(v3,v7)) {
            tryWedge(v,{1,5,4},{2,6,7}, numUniqueVertices);
            continue;
            // return;
          }

          // check right side:
          if (same(v1,v2) && same(v5,v6)) {
            tryWedge(v,{7,4,5},{3,0,1}, numUniqueVertices);
            continue;
            // return;
          }
          if (same(v1,v5) && same(v2,v6)) {
            tryWedge(v,{4,0,1},{7,3,2}, numUniqueVertices);
            continue;
            // return;
          }
          
          // ==================================================================
          // fallback - there's still cases of only ONE collapsed vertex,
          // for example, so let's just make this into a deformed hex 
          // ==================================================================
          emitHex(v,/*perfect:*/false);
          continue;
          // return;
        }
  }
  
  
  void process(Exa &exa)
  {
    std::cout << "sorting cell list for query" << std::endl;
    std::sort(exa.cellList.begin(),exa.cellList.end());
    std::cout << "Sorted .... starting to query" << std::endl;
#if DEBUG
    serial_for
#else
      parallel_for
#endif
      (exa.cellList.size(),
       [&](size_t cellID){
         // PING; PRINT(cellID);
         const Exa::Cell &cell = exa.cellList[cellID];
         doCell(exa,cell);
       });
  }




  extern "C" int main(int ac, char **av)
  {
    std::string cellsFileName = "";
    std::string scalarsFileName = "";
    std::string outFileName = "";
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-o")
        outFileName = av[++i];
      else if (arg == "--boundary-only")
        boundaryOnly = true;
      else if (arg[0] == '-')
        throw std::runtime_error("./exa2umesh in.cells scalar.scalars -o out.umesh [--boundary-only]\n");
      else {
        if (cellsFileName == "")
          cellsFileName = arg;
        else if (scalarsFileName == "")
          scalarsFileName = arg;
        else if (outFileName == "")
          outFileName = arg;
        else 
          throw std::runtime_error("./exa2umesh in.cells scalar.scalars -o out.umesh [--boundary-only]\n");
      }
    }
    cout.precision(10);
    Exa exa;
    std::ifstream in_cells(cellsFileName);
    std::ifstream in_scalars(scalarsFileName);
    output = std::make_shared<UMesh>();
  
    while (!in_cells.eof()) {
      Exa::Cell cell;
      in_cells.read((char*)&cell,sizeof(Exa::LogicalCell));
      in_scalars.read((char*)&cell.scalar,sizeof(float));
    
      if (!(in_cells.good() && in_scalars.good()))
        break;
    
      exa.add(cell);
      // static size_t nextPing = 1;
      // if (exa.size() >= nextPing) {
      //   std::cout << "read so far : " << exa.size() << ", bounds " << exa.bounds << std::endl;
      //   nextPing *= 2;
      // }
    }
    std::cout << "done reading, found " << prettyNumber(exa.size()) << " cells" << std::endl;

    output->perVertex = std::make_shared<Attribute>();
    
    process(exa);

    output->finalize();
    std::cout << "created umesh " << output->toString() << std::endl;
    std::cout << "running sanity checks:" << std::endl;
    sanityCheck(output);
    //io::saveBinaryUMesh(outFileName,output);
    std::cout << "saving to " << outFileName << std::endl;

    PRINT(output->vertices.size());
    PRINT(output->hexes.size());
    output->saveTo(outFileName);

#if 1
    {
      UMesh::SP tmp = std::make_shared<UMesh>();
      tmp->vertices = output->vertices;
      tmp->perVertex = output->perVertex;
      tmp->hexes = output->hexes;
      tmp->saveTo(outFileName+"_hexes.umesh");
    }
#endif

#if 1
    {
      UMesh::SP tmp = std::make_shared<UMesh>();
      tmp->vertices = output->vertices;
      tmp->perVertex = output->perVertex;
      tmp->pyrs = output->pyrs;
      tmp->saveTo(outFileName+"_pyrs.umesh");
    }
#endif

#if 1
    {
      UMesh::SP tmp = std::make_shared<UMesh>();
      tmp->vertices = output->vertices;
      tmp->perVertex = output->perVertex;
      tmp->wedges = output->wedges;
      tmp->saveTo(outFileName+"_wedges.umesh");
    }
#endif

#if 1
    {
      UMesh::SP tmp = std::make_shared<UMesh>();
      tmp->vertices = output->vertices;
      tmp->perVertex = output->perVertex;
      tmp->tets = output->tets;
      tmp->saveTo(outFileName+"_tets.umesh");
    }
#endif
  }

}
