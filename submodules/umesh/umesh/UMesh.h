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

/*! how to use 'UMESH_EXTERNAL_MATH': by default (ie, if
    UMESH_EXTERNAL_MESH is either false or not defined, umesh will use
    its own math classes umesh::vec3f, etc, from umesh/math.h, as well
    as its own umesh::parallel_for from umesh/parallel_for.h. If,
    however, you want umesh to use your vec3f, box3f, vec3i, etc
    classes, (eg, from owl::common namespace, then you can, instead,
    before including any umesh classes, to the following 

    namespace umesh {
    using namespace owl;
    }
    #include <umesh/UMesh.h>
*/
#ifdef UMESH_EXTERNAL_MATH
/* to not include out own math, assume that somebody outside this
   project has defined the umesh::vec3f, umesh::vec3i, and
   umesh::vec4i types */
#else
# include "umesh/math.h"
# include "umesh/parallel_for.h"
#endif
#include <mutex>
#include <vector>
#include <map>
#include <assert.h>
#include <memory>
#include <algorithm>
#include <string>

#ifdef _MSC_VER
# define UMESH_ALIGN(alignment) __declspec(align(alignment)) 
#else
# define UMESH_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif
#define UMESH_TERMINAL_RED "\033[0;31m"
#define UMESH_TERMINAL_GREEN "\033[0;32m"
#define UMESH_TERMINAL_LIGHT_GREEN "\033[1;32m"
#define UMESH_TERMINAL_YELLOW "\033[1;33m"
#define UMESH_TERMINAL_BLUE "\033[0;34m"
#define UMESH_TERMINAL_LIGHT_BLUE "\033[1;34m"
#define UMESH_TERMINAL_RESET "\033[0m"
#define UMESH_TERMINAL_DEFAULT UMESH_TERMINAL_RESET
#define UMESH_TERMINAL_BOLD "\033[1;1m"

#define UMESH_TERMINAL_MAGENTA "\e[35m"
#define UMESH_TERMINAL_LIGHT_MAGENTA "\e[95m"
#define UMESH_TERMINAL_CYAN "\e[36m"
//#define UMESH_TERMINAL_LIGHT_RED "\e[91m"
#define UMESH_TERMINAL_LIGHT_RED "\033[1;31m"

#if (!defined(__umesh_both__))
# if defined(__CUDA_ARCH__)
#  define __umesh_device   __device__
#  define __umesh_host     __host__
# else
#  define __umesh_device   /* ignore */
#  define __umesh_host     /* ignore */
# endif
# define __umesh_both__   __umesh_host __umesh_device
#endif
  
namespace umesh {

  /*! can be used to turn on/off logging/diagnostic messages in entire
    umesh library */
  extern bool verbose;
  
  struct Attribute {
    typedef std::shared_ptr<Attribute> SP;

    Attribute(int num=0) : values(num) {}
    
    /*! tells this attribute that its values are set, and precomputations can be done */
    void finalize();
    
    std::string name;
    /*! for now lets implement only float attributes. node/ele files
      actually seem to have ints in the ele file (if they have
      anything at all), but these should still fir into floats, so
      we should be good for now. This may change once we better
      understand where these values come from, and what they
      mean. */
    std::vector<float> values;
    range1f valueRange;
  };

  struct Triangle {
    enum { numVertices = 3 };
    inline Triangle() = default;
    inline Triangle(const Triangle &) = default;
    inline Triangle(int v0, int v1, int v2)
      : x(v0), y(v1), z(v2)
    {}
    inline Triangle(const vec3i &v)
      : x(v.x), y(v.y), z(v.z)
    {}

    inline operator vec3i() const { return vec3i(x,y,z); }

    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    int x,y,z;
  };

  struct Quad {
    enum { numVertices = 4 };
    inline Quad() = default;
    inline Quad(const Quad &) = default;
    inline Quad(int v0, int v1, int v2, int v3)
      : x(v0), y(v1), z(v2), w(v3)
    {}
    inline Quad(const vec4i &v)
      : x(v.x), y(v.y), z(v.z), w(v.w)
    {}

    inline operator vec4i() const { return vec4i(x,y,z,w); }
    
    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    int x,y,z,w;
  };

  struct Tet {
    enum { numVertices = 4 };
    inline Tet() = default;
    inline Tet(const Tet &) = default;
    inline Tet(int v0, int v1, int v2, int v3)
      : x(v0), y(v1), z(v2), w(v3)
    {}
    inline Tet(const vec4i &v)
      : x(v.x), y(v.y), z(v.z), w(v.w)
    {}

    inline operator vec4i() const { return vec4i(x,y,z,w); }
    
    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    int x,y,z,w;
  };

  struct Pyr {
    enum { numVertices = 5 };
    inline Pyr() = default;
    inline Pyr(const Pyr &) = default;
    inline Pyr(int v0, int v1, int v2, int v3, int v4)
      : base(v0,v1,v2,v3),top(v4)
    {}
      
    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    vec4i base;
    int top;
  };
    
  inline std::ostream &operator<<(std::ostream& out, Pyr p) {
    out << '(' << p.base << ',' << p.top << ')';
    return out;
  }

  struct Wedge {
    enum { numVertices = 6 };
    inline Wedge() {}
    inline Wedge(int v0, int v1, int v2, int v3, int v4, int v5)
      : front(v0,v1,v2),
        back(v3,v4,v5)
    {}
    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }  
    vec3i front, back;
  };

  inline std::ostream &operator<<(std::ostream& out, Wedge w) {
    out << '(' << w.front << ',' << w.back << ')';
    return out;
  }

  struct Hex {
    enum { numVertices = 8 };
    inline Hex() {}
    inline Hex(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7)
      : base(v0,v1,v2,v3),
        top(v4,v5,v6,v7)
    {}
    /*! array operator, assuming VTK ordering (see windingOrder.png
      file for illustration) */
    inline const int &operator[](int i) const {
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    inline int &operator[](int i){
      assert(i>=0 && i<numVertices);
      return ((int*)this)[i];
    }
    vec4i base;
    vec4i top;
  };
    
  inline std::ostream &operator<<(std::ostream& out, Hex h) {
    out << '(' << h.base << ',' << h.top << ')';
    return out;
  }
    
  /*! basic unstructured mesh class - one set of 3-float vertices, and
    one std::vector each for tets, wedges, pyramids, and hexes, all
    using VTK format for the vertex index ordering (see
    windingorder.jpg) */
  struct UMesh {
    using Triangle = umesh::Triangle;
    using Quad     = umesh::Quad;
    using Tet      = umesh::Tet;
    using Pyr      = umesh::Pyr;
    using Wedge    = umesh::Wedge;
    using Hex      = umesh::Hex;
    
    typedef std::shared_ptr<UMesh> SP;

    typedef enum { TRI, QUAD, TET, PYR, WEDGE, HEX, INVALID } PrimType;
    
    struct PrimRef {
      inline PrimRef() {}
      inline PrimRef(PrimType type, size_t ID) : type(type), ID(ID) {}
      inline PrimRef(const PrimRef &) = default;
      inline bool isTet() const { return type == TET; }
      union {
        struct {
          size_t type:4;
          size_t ID:60;
        };
        size_t as_size_t;
      };
    };

    /*! return a string of the form "UMesh{#tris=...}" */
    std::string toString(bool compact=true) const;
    
    /*! returns total numer of volume elements */
    inline size_t numVolumeElements() const
    {
      return tets.size()+pyrs.size()+wedges.size()+hexes.size();
    }
    
    /*! print some basic info of this mesh to std::cout */
    void print();

    /*! write - binary - to given file */
    void saveTo(const std::string &fileName) const;
    /*! write - binary - to given (bianry) stream */
    void writeTo(std::ostream &out) const;
    
    
    /*! read from given file, assuming file format as used by saveTo() */
    static UMesh::SP loadFrom(const std::string &fileName);
    /*! read from given (binary) stream */
    void readFrom(std::istream &in);
    
    
    /*! create std::vector of primitmive references (bounding box plus
      tag) for every volumetric prim in this mesh */
    void createVolumePrimRefs(std::vector<PrimRef> &result);

    /*! create std::vector of primitmive references (bounding box plus
      tag) for every volumetric prim in this mesh */
    std::vector<PrimRef> createVolumePrimRefs();

    /*! create std::vector of all primrefs for all _surface_ elements
        (triangles and quads) */
    void createSurfacePrimRefs(std::vector<PrimRef> &result);
    
    /*! create std::vector of all primrefs for all _surface_ elements
        (triangles and quads) */
    std::vector<PrimRef> createSurfacePrimRefs();
    
    inline size_t size() const
    {
      return
        triangles.size()+
        quads.size()+
        hexes.size()+
        tets.size()+
        wedges.size()+
        pyrs.size();
    }

    inline range1f getValueRange() const {
      // if (perVertex) {
      if (perVertex)
        return perVertex->valueRange;
      // } else {
      //   range1f result;
      //   if (perTet)   result.extend(perTet->valueRange);
      //   if (perHex)   result.extend(perHex->valueRange);
      // }
      throw std::runtime_error("cannot get value range for umesh: no attributes!");
    }
    
    inline box3f getBounds() const
    {
      return bounds;
    }

    inline box4f getBounds4f() const
    {
      return box4f(vec4f(bounds.lower,perVertex->valueRange.lower),
                   vec4f(bounds.upper,perVertex->valueRange.upper));
    }
    



    inline range1f getTetValueRange(const size_t ID) const
    {
      assert(ID < tets.size());
      const range1f b = range1f()
        .including(perVertex->values[tets[ID].x])
        .including(perVertex->values[tets[ID].y])
        .including(perVertex->values[tets[ID].z])
        .including(perVertex->values[tets[ID].w]);
      return b;
    }
    
    inline range1f getPyrValueRange(const size_t ID) const
    {
      assert(ID < pyrs.size());
      return range1f()
        .including(perVertex->values[pyrs[ID].top])
        .including(perVertex->values[pyrs[ID].base.x])
        .including(perVertex->values[pyrs[ID].base.y])
        .including(perVertex->values[pyrs[ID].base.z])
        .including(perVertex->values[pyrs[ID].base.w]);
    }
    
    inline range1f getWedgeValueRange(const size_t ID) const
    {
      assert(ID < wedges.size());
      return range1f()
        .including(perVertex->values[wedges[ID].front.x])
        .including(perVertex->values[wedges[ID].front.y])
        .including(perVertex->values[wedges[ID].front.z])
        .including(perVertex->values[wedges[ID].back.x])
        .including(perVertex->values[wedges[ID].back.y])
        .including(perVertex->values[wedges[ID].back.z])
        ;
    }
    
    inline range1f getHexValueRange(const size_t ID) const
    {
      assert(ID < hexes.size());
      const range1f b = range1f()
        .including(perVertex->values[hexes[ID].top.x])
        .including(perVertex->values[hexes[ID].top.y])
        .including(perVertex->values[hexes[ID].top.z])
        .including(perVertex->values[hexes[ID].top.w])
        .including(perVertex->values[hexes[ID].base.x])
        .including(perVertex->values[hexes[ID].base.y])
        .including(perVertex->values[hexes[ID].base.z])
        .including(perVertex->values[hexes[ID].base.w]);
      return b;
    }
    
    inline range1f getValueRange(const PrimRef &pr) const
    {
      switch(pr.type) {
      case TET:  return getTetValueRange(pr.ID);
      case PYR:  return getPyrValueRange(pr.ID);
      case WEDGE:return getWedgeValueRange(pr.ID);
      case HEX:  return getHexValueRange(pr.ID);
      default: 
        throw std::runtime_error("not implemented");
      };
    }



    
    

    inline box3f getTetBounds(const size_t ID) const
    {
      assert(ID < tets.size());
      const box3f b = box3f()
        .including(vertices[tets[ID].x])
        .including(vertices[tets[ID].y])
        .including(vertices[tets[ID].z])
        .including(vertices[tets[ID].w]);
      return b;
    }
    
    inline box3f getPyrBounds(const size_t ID) const
    {
      assert(ID < pyrs.size());
      return box3f()
        .including(vertices[pyrs[ID].top])
        .including(vertices[pyrs[ID].base.x])
        .including(vertices[pyrs[ID].base.y])
        .including(vertices[pyrs[ID].base.z])
        .including(vertices[pyrs[ID].base.w]);
    }
    
    inline box3f getWedgeBounds(const size_t ID) const
    {
      assert(ID < wedges.size());
      return box3f()
        .including(vertices[wedges[ID].front.x])
        .including(vertices[wedges[ID].front.y])
        .including(vertices[wedges[ID].front.z])
        .including(vertices[wedges[ID].back.x])
        .including(vertices[wedges[ID].back.y])
        .including(vertices[wedges[ID].back.z]);
    }
    
    inline box3f getHexBounds(const size_t ID) const
    {
      assert(ID < hexes.size());
      const box3f b = box3f()
        .including(vertices[hexes[ID].top.x])
        .including(vertices[hexes[ID].top.y])
        .including(vertices[hexes[ID].top.z])
        .including(vertices[hexes[ID].top.w])
        .including(vertices[hexes[ID].base.x])
        .including(vertices[hexes[ID].base.y])
        .including(vertices[hexes[ID].base.z])
        .including(vertices[hexes[ID].base.w]);
      return b;
    }
    
    inline box3f getBounds(const PrimRef &pr) const
    {
      switch(pr.type) {
      case TET:  return getTetBounds(pr.ID);
      case PYR:  return getPyrBounds(pr.ID);
      case WEDGE: return getWedgeBounds(pr.ID);
      case HEX:  return getHexBounds(pr.ID);
      default: 
        throw std::runtime_error("not implemented");
      };
    }

    inline box4f getBounds4f(const PrimRef &pr) const
    {
      box3f   spatial = getBounds(pr);
      range1f value   = getValueRange(pr);
      return box4f(vec4f(spatial.lower,value.lower),
                   vec4f(spatial.upper,value.upper));
    }

    /*! finalize a mesh, and compute min/max ranges where required */
    void finalize();
    
    std::vector<vec3f> vertices;
    Attribute::SP      perVertex;
    // Attribute::SP      perTet;
    // Attribute::SP      perHex;
    
    // -------------------------------------------------------
    // surface elements:
    // -------------------------------------------------------
    std::vector<Triangle> triangles;
    std::vector<Quad>     quads;

    // -------------------------------------------------------
    // volume elements:
    // -------------------------------------------------------
    std::vector<Tet>   tets;
    std::vector<Pyr>   pyrs;
    std::vector<Wedge> wedges;
    std::vector<Hex>   hexes;

    /*! in some cases, it makes sense to allow for storing a
      user-provided per-vertex tag (may be empty) */
    std::vector<size_t> vertexTag;
    
    box3f   bounds;
  };
  
  /*! helper functoin for printf debugging - puts the four elemnt
      sizes into a human-readable (short) string*/
  std::string sizeString(UMesh::SP mesh);

} // ::umesh
