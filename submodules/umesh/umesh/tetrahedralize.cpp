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

#include "umesh/tetrahedralize.h"
#include <algorithm>
#include <set>

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
  struct MergedMesh {
    
    MergedMesh(UMesh::SP in, bool passThroughFlatElements=false)
      : in(in),
        out(std::make_shared<UMesh>()),
        passThroughFlatElements(passThroughFlatElements)
    {
      out->vertices = in->vertices;
      if (in->perVertex) {
        out->perVertex = std::make_shared<Attribute>();
        out->perVertex->name   = in->perVertex->name;
        out->perVertex->values = in->perVertex->values;
      }
    }

    inline float length(const vec3f v) { return sqrtf(dot(v,v)); }
    
    inline float volume(const vec3f &v0,
                        const vec3f &v1,
                        const vec3f &v2,
                        const vec3f &v3)
    {
      return dot(v3-v0,cross(v1-v0,v2-v0));
    }
                      
    inline bool flat(const vec3f &v0,
                     const vec3f &v1,
                     const vec3f &v2,
                     const vec3f &v3)
    {
      if (v0 == v1 ||
          v0 == v2 ||
          v0 == v3 ||
          v1 == v2 ||
          v1 == v3 ||
          v2 == v3)
        return false;
      const vec3f n0 = cross(v1-v0,v2-v0);
      if (length(n0) == 0.f) return false;
      
      const vec3f n1 = cross(v2-v0,v3-v0);
      if (length(n1) == 0.f) return false;

      return dot(n0,n1)/(length(n0)*length(n1)) >= .99f;
    }
                      
    
    void add(const UMesh::Tet &tet, const std::string &dbg="")
    {
      if (tet.x == tet.y ||
          tet.x == tet.z ||
          tet.x == tet.w ||
          tet.y == tet.z ||
          tet.y == tet.w ||
          tet.z == tet.w)
        /* degenerate/flat tet .... so in either case: dump this */
        return;
      vec3f a = out->vertices[tet.x];
      vec3f b = out->vertices[tet.y];
      vec3f c = out->vertices[tet.z];
      vec3f d = out->vertices[tet.w];
      float volume = dot(d-a,cross(b-a,c-a));
      
      if (volume == 0.f)
        /* degenerate/flat - dump this */
        return;

      if (volume < 0.f) {
        static bool warned = false;
        if (!warned) {
          std::cout
            << UMESH_TERMINAL_RED
            <<"WARNING: at least one tet (or other element that generated a tet)\n was wrongly oriented!!! (I'll swap those tets, but that's still fishy...)"
            << UMESH_TERMINAL_DEFAULT
            << std::endl;
          warned = true;
        }
        out->tets.push_back({tet.x,tet.y,tet.w,tet.z});
      } else
        out->tets.push_back(tet);
    }

    void add(const UMesh::Pyr &pyr, const std::string &dbg="")
    {
      if (passThroughFlatElements) {
        const vec3f v0 = out->vertices[pyr[0]];
        const vec3f v1 = out->vertices[pyr[1]];
        const vec3f v2 = out->vertices[pyr[2]];
        const vec3f v3 = out->vertices[pyr[3]];
        const vec3f v4 = out->vertices[pyr[4]];
        if (flat(v0,v1,v2,v3)) {
          if (volume(v0,v1,v2,v4) < 0.f) {
            UMesh::Pyr _pyr = pyr;
            std::swap(_pyr.base.x,_pyr.base.y);
            std::swap(_pyr.base.z,_pyr.base.w);
            out->pyrs.push_back(_pyr);
          } else 
            out->pyrs.push_back(pyr);
          // passed this through; done.
          return;
        }
      }
      int base = getCenter({pyr[0],pyr[1],pyr[2],pyr[3]});
      add(UMesh::Tet(pyr[0],pyr[1],base,pyr[4]),dbg+"pyr0");
      add(UMesh::Tet(pyr[1],pyr[2],base,pyr[4]),dbg+"pyr1");
      add(UMesh::Tet(pyr[2],pyr[3],base,pyr[4]),dbg+"pyr2");
      add(UMesh::Tet(pyr[3],pyr[0],base,pyr[4]),dbg+"pyr3");
    }

    void add(const UMesh::Wedge &wedge)
    {
      const vec3f v0 = out->vertices[wedge[0]];
      const vec3f v1 = out->vertices[wedge[1]];
      const vec3f v2 = out->vertices[wedge[2]];
      const vec3f v3 = out->vertices[wedge[3]];
      const vec3f v4 = out->vertices[wedge[4]];
      const vec3f v5 = out->vertices[wedge[5]];
      if (v2 == v5)
        throw std::runtime_error("wedge that should be a pyramid!?");
      
      if (passThroughFlatElements) {
        if (flat(v0,v2,v5,v3) &&
            flat(v1,v2,v5,v4) &&
            flat(v0,v1,v4,v3)) {
          if (volume(v0,v1,v4,v2) < 0.f) {
            UMesh::Wedge _wedge = wedge;
            std::swap(_wedge[0],_wedge[3]);
            std::swap(_wedge[1],_wedge[4]);
            std::swap(_wedge[2],_wedge[5]);
            out->wedges.push_back(_wedge);
          } else 
            out->wedges.push_back(wedge);
          // passed this through; done.
          return;
        }
      }
      
      std::set<vec3f> uniqueBaseVertices;
      uniqueBaseVertices.insert(v0);
      uniqueBaseVertices.insert(v1);
      uniqueBaseVertices.insert(v3);
      uniqueBaseVertices.insert(v4);
      if (uniqueBaseVertices.size() == 4) {
        // newly created points:
        int center = getCenter({wedge[0],wedge[1],wedge[2],
                                wedge[3],wedge[4],wedge[5]});
        
        // bottom face to center
        add(UMesh::Pyr(wedge[0],wedge[1],wedge[4],wedge[3],center),"wed1");
        // left face to center
        add(UMesh::Pyr(wedge[0],wedge[3],wedge[5],wedge[2],center),"wed2");
        // right face to center
        add(UMesh::Pyr(wedge[1],wedge[2],wedge[5],wedge[4],center),"wed3");
        // front face to center
        add(UMesh::Tet(wedge[0],wedge[2],wedge[1],center),"wed4");
        // back face to center
        add(UMesh::Tet(wedge[3],wedge[4],wedge[5],center),"wed5");
      } else if (uniqueBaseVertices.size() == 3) {
        if (v0 == v1) {
          int center = getCenter({wedge[0],wedge[2],
                                  wedge[3],wedge[4],wedge[5]});
          // bottom face to center
          add(UMesh::Tet(wedge[0],wedge[4],wedge[3],center),"wed1");
          // left face to center
          add(UMesh::Pyr(wedge[0],wedge[3],wedge[5],wedge[2],center),"wed2");
          // right face to center
          add(UMesh::Pyr(wedge[1],wedge[2],wedge[5],wedge[4],center),"wed3");
          // // front face to center
          // add(UMesh::Tet(wedge[0],wedge[2],wedge[1],center),"wed4");
          // back face to center
          add(UMesh::Tet(wedge[3],wedge[4],wedge[5],center),"wed5");
          
        } else if (v3 == v4) {
          int center = getCenter({wedge[0],wedge[1],wedge[2],
                                  wedge[3],wedge[5]});
          // bottom face to center
          add(UMesh::Tet(wedge[0],wedge[1],wedge[3],center),"wed1");
          // left face to center
          add(UMesh::Pyr(wedge[0],wedge[3],wedge[5],wedge[2],center),"wed2");
          // right face to center
          add(UMesh::Pyr(wedge[1],wedge[2],wedge[5],wedge[4],center),"wed3");
           // front face to center
          add(UMesh::Tet(wedge[0],wedge[2],wedge[1],center),"wed4");
          // // back face to center
          // add(UMesh::Tet(wedge[3],wedge[4],wedge[5],center),"wed5");
        } else
          throw std::runtime_error("oy-wey.... what _is_ that shape!?");
      } else {
        throw std::runtime_error("wedge that should be a tet!?");
      }
    }
    
    void add(const UMesh::Hex &hex)
    {
      if (passThroughFlatElements) {
        const vec3f v0 = out->vertices[hex[0]];
        const vec3f v1 = out->vertices[hex[1]];
        const vec3f v2 = out->vertices[hex[2]];
        const vec3f v3 = out->vertices[hex[3]];
        const vec3f v4 = out->vertices[hex[4]];
        const vec3f v5 = out->vertices[hex[5]];
        const vec3f v6 = out->vertices[hex[6]];
        const vec3f v7 = out->vertices[hex[7]];
        if (flat(v0,v1,v2,v3) &&
            flat(v4,v5,v6,v7) &&
            flat(v1,v2,v6,v5) &&
            flat(v0,v3,v7,v4) &&
            flat(v0,v1,v5,v4) &&
            flat(v3,v2,v6,v7)) {
          if (volume(v0,v1,v2,v5) < 0.f) {
            UMesh::Hex _hex = hex;
            std::swap(_hex[0],_hex[4]);
            std::swap(_hex[1],_hex[5]);
            std::swap(_hex[2],_hex[6]);
            std::swap(_hex[3],_hex[7]);
            out->hexes.push_back(_hex);
          } else 
            out->hexes.push_back(hex);
          // passed this through; done.
          for (int i=0;i<8;i++)
            assert(out->hexes.back()[i] >= 0);
          return;
        }
      }
      // newly created points:
      int center = getCenter({hex[0],hex[1],hex[2],hex[3],
                              hex[4],hex[5],hex[6],hex[7]});

      // bottom face to center
      add(UMesh::Pyr(hex[0],hex[1],hex[2],hex[3],center));
      // top face to center
      add(UMesh::Pyr(hex[4],hex[7],hex[6],hex[5],center));

      // front face to center
      add(UMesh::Pyr(hex[0],hex[4],hex[5],hex[1],center));
      // back face to center
      add(UMesh::Pyr(hex[2],hex[6],hex[7],hex[3],center));

      // left face to center
      add(UMesh::Pyr(hex[0],hex[3],hex[7],hex[4],center));
      // right face to center
      add(UMesh::Pyr(hex[1],hex[5],hex[6],hex[2],center));
    }
    
    int getCenter(std::vector<int> idx)
    {
      std::sort(idx.begin(),idx.end());
      auto it = newVertices.find(idx);
      if (it != newVertices.end())
        return it->second;
      
      vec3f centerPos = vec3f(0.f);
      float centerVal = 0.f;
      for (auto i : idx) {
        if (in->perVertex)
          centerVal += in->perVertex->values[i];
        centerPos = centerPos + in->vertices[i];
      }
      centerVal *= (1.f/idx.size());
      centerPos = centerPos * (1.f/idx.size());

      // now let's be super-pedantic, and check if the newly generated
      // vertex by pure chance already exists in the input. shouldn't
      // happen, but who knows...
      int ID = -1;
      auto it2 = vertices.find(centerPos);
      if (it2 != vertices.end())
        ID = it2->second;
      else {
        ID = (int)out->vertices.size();
        out->vertices.push_back(centerPos);
        if (out->perVertex)
          out->perVertex->values.push_back(centerVal);
      }
      newVertices[idx] = ID;
      return ID;
    }

    std::map<vec3f,int> vertices;
    std::map<std::vector<int>,int> newVertices;
    UMesh::SP in, out;
    /*! if true, then we'll tessellate only curved elements */
    bool passThroughFlatElements;
  };



#if 0
  // only use for debugging, to force priting of prims that contain certain vertices or faces

  template<typename T>
  inline bool contains(UMesh::SP in, T t, int ID)
  {
    for (int i=0;i<t.numVertices;i++)
      if (t[i] == ID) return true;
    return false;
  }
  
  template<typename T>
  inline bool contains(UMesh::SP in, T t, const vec3f &pos)
  {
    for (int i=0;i<t.numVertices;i++)
      if (in->vertices[t[i]] == pos) return true;
    return false;
  }
  
  template<typename T>
  inline bool offending(UMesh::SP in, T t)
  {
    return
      contains(in,t,vec3f(4.5f,1.5f,1.5f)) &&
      contains(in,t,vec3f(5.5f,1.5f,1.5f)) ;
      // contains(t,830) &&
      // contains(t,858);
  }
#endif
  




  
  /*! tetrahedralize a given input mesh; in a way that non-triangular
    faces (from wedges, pyramids, and hexes) will always be
    subdividied exactly the same way even if that same face is used
    by another element with different vertex order
  
    Notes:

    - tetrahedralization will create new vertices; if the input
    contained a scalar field the output will contain the same
    scalars for existing vertices, and will interpolate for newly
    created ones.
      
    - output will *not* contain the input's surface elements
    
    - the input's vertices will be fully contained within the output's
    vertex array, with the same indices; so any surface elements
    defined in the input should remain valid in the output's vertex
    array.
  */
  UMesh::SP tetrahedralize(UMesh::SP in)
  {
#if 0
    // for (auto vtx : in->vertices)
    //   PRINT(vtx);
    PING;
    for (auto prim : in->tets)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->pyrs)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->wedges)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->hexes)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
#endif    

    MergedMesh merged(in);
    for (auto tet : in->tets)
      merged.add(tet);
    for (auto pyr : in->pyrs)
      merged.add(pyr);
    for (auto wedge : in->wedges) 
      merged.add(wedge);
    for (auto hex : in->hexes) 
      merged.add(hex);
    std::cout << "done tetrahedralizing, got "
              << sizeString(merged.out)
              << " from " << sizeString(in) << std::endl;
    return merged.out;
  }


  /*! same as tetrahedralize(umesh), but will, eventually, contain
      only the tetrahedra created by the 'owned' elements listed, EVEN
      THOUGH the vertex array produced by that will be the same as in
      the original tetrahedralize(mesh) version */
  UMesh::SP tetrahedralize(UMesh::SP in,
                           int ownedTets,
                           int ownedPyrs,
                           int ownedWedges,
                           int ownedHexes)
  {
    // ------------------------------------------------------------------
    // first stage: push _all_ elements, to ensure we get same vertex
    // array as 'non-owned' version
    // ------------------------------------------------------------------
    MergedMesh merged(in);
    for (auto tet : in->tets)
      merged.add(tet);
    for (auto pyr : in->pyrs)
      merged.add(pyr);
    for (auto wedge : in->wedges) 
      merged.add(wedge);
    for (auto hex : in->hexes) 
      merged.add(hex);
    // ------------------------------------------------------------------
    // now, kll all so-far generated tets in the output mesh, but keep
    // the vertex array...
    // ------------------------------------------------------------------
    merged.out->tets.clear();
    // ------------------------------------------------------------------
    // and re-push (only) the owned ones
    // ------------------------------------------------------------------
    for (size_t i=0;i<std::min((int)in->tets.size(),ownedTets);i++)
      merged.add(in->tets[i]);
    for (size_t i=0;i<std::min((int)in->pyrs.size(),ownedPyrs);i++)
      merged.add(in->pyrs[i]);
    for (size_t i=0;i<std::min((int)in->wedges.size(),ownedWedges);i++)
      merged.add(in->wedges[i]);
    for (size_t i=0;i<std::min((int)in->hexes.size(),ownedHexes);i++)
      merged.add(in->hexes[i]);
    std::cout << "finalizing..." << std::endl;
    merged.out->finalize();
    std::cout << "done tetrahedralizing (second stage), got "
              << sizeString(merged.out)
              << " from " << sizeString(in) << std::endl;
    return merged.out;
  }

  /*! same as tetrahedralize(), but chop up ONLY elements with curved
      sides, and pass through all those that have flat sides. */
  UMesh::SP tetrahedralize_maintainFlatElements(UMesh::SP in)
  {
#if 0
    PING;
    for (auto prim : in->tets)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->pyrs)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->wedges)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
    for (auto prim : in->hexes)
      if (offending(in,prim)) {
        PRINT(prim);
        for (int i=0;i<prim.numVertices;i++)
          PRINT(in->vertices[prim[i]]);
      }
#endif    

    MergedMesh merged(in,/*pass through flat elements:*/true);
    for (auto tet : in->tets)
      merged.add(tet);
    for (auto pyr : in->pyrs)
      merged.add(pyr);
    for (auto wedge : in->wedges) 
      merged.add(wedge);
    for (auto hex : in->hexes) 
      merged.add(hex);
    std::cout << "finalizing..." << std::endl;
    merged.out->finalize();
    std::cout << "done tetrahedralizing curved elements (pass through for flat), got "
              << sizeString(merged.out)
              << " from " << sizeString(in) << std::endl;
    return merged.out;
  }
  
} // ::umesh
