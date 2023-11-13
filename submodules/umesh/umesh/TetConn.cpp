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

#include "TetConn.h"
#include "umesh/io/IO.h"
#include <fstream>

namespace umesh {

  /*! helper class that maintains all temporary helper data structures */
  struct TetConnHelper
  {
    /*! constructor will compute the given 'out' struct, from given 'in' mesh */
    TetConnHelper(TetConn &out, const UMesh &in);

  private:
    /*! push facet #facetIdx of tet #tetIdx, with given vertex indices
      (oriented to point *towards* tet the given tet */
    void pushFacet(int tetIdx, int facetIdx, vec3i indices);
    /*! push tet #tetIdx from input mesh in, and fill in all respective
      values in the 'out' field */
    void pushTet(int tetIdx);

    /*! find (or allocate) face that shars these indices (in any
      order), and return Idx in 'perFace' vectors, as well as whether
      given indices refer to the front- or back-side of this found
      face */
    int findFace(vec3i indices, int &side);
    
    const UMesh &in;
    TetConn &out;
    
    std::map<vec3i,int> faceIndex;
  };
  
  TetConnHelper::TetConnHelper(TetConn &out,
                               const UMesh &in)
    : out(out), in(in)
  {
    if (!in.wedges.empty() ||
        !in.pyrs.empty() ||
        !in.hexes.empty())
      throw std::runtime_error
        ("cowardly refusing to compute tet-connectivity on a mesh "
         "that contains non-tet elements .... ");
    
    if (in.vertices.size() >= (1ull<<31))
      throw std::runtime_error("number of input vertices too large - would overflow");
    if (in.tets.size() >= (1ull<<31))
      throw std::runtime_error("number of input vertices too large - would overflow");
      
    out.faces.clear();
      
    out.tetFaces.clear();
      
    for (size_t i=0;i<in.tets.size();i++)
      pushTet((int)i);
  }

  /*! find (or allocate) face that shars these indices (in any
    order), and return Idx in 'perFace' vectors, as well as whether
    given indices refer to the front- or back-side of this found
    face */
  int TetConnHelper::findFace(vec3i indices, int &side)
  {
    side = 0;
      
    /* first - compute unique indexing key (by sorting), and
       orientation of input face relative to that indexing */
    for (int i=0;i<3;i++) 
      for (int j=0;j<i;j++) 
        if (indices[i] < indices[j]) {
          std::swap(indices[i],indices[j]);
          side = 1-side;
        }

    if (indices.x >= indices.y || indices.x >= indices.z
        || indices.y > indices.z)
      throw std::runtime_error("not sorted indices!?");
    
    /* now see if we have this face already */
    auto it = faceIndex.find(indices);
    if (it != faceIndex.end()) {
      if (out.faces[it->second].index != indices)
        throw std::runtime_error("wrong face index mapping!?");
      /* ... and if so, return it */
      return it->second;
    }

    /* ... and if not, allocate ... */
    if (out.faces.size() >= (1ull<<31))
      throw std::runtime_error
        ("too many faces - can't index with 32-bit (signed) ints");

    int newIdx = int(out.faces.size());
    /* ... and add */
    TetConn::Face newFace;
    newFace.index = { indices };
    out.faces.push_back(newFace);
    faceIndex[indices] = newIdx;
    return newIdx;
  }

  float volume(const std::vector<vec3f> &pos,
               const vec4i &idx)
  {
    const vec3f a = pos[idx.x];
    const vec3f b = pos[idx.y];
    const vec3f c = pos[idx.z];
    const vec3f d = pos[idx.w];
    return dot(d-a,cross(b-a,c-a));
  }
  
  /*! push facet #facetIdx of tet #tetIdx, with given vertex indices
    (oriented to point *towards* tet the given tet */
  void TetConnHelper::pushFacet(int tetIdx,
                                int facetIdx,
                                vec3i indices)
  {
    int side;
    int faceIdx = findFace(indices,side);
    TetConn::Face &face = out.faces[faceIdx];
    if (face.tetIdx[side] != -1) {
      // PRINT(indices);
      // PRINT(faceIdx);
      // PRINT(side);
      // PRINT(in.tets[face.tetIdx[side]]);
      // PRINT(volume(in.vertices,in.tets[face.tetIdx[side]]));
      // PRINT(volume(in.vertices,in.tets[tetIdx]));
      // PRINT(face.tetIdx[0]);
      // PRINT(face.tetIdx[1]);
      // PRINT(in.vertices[in.tets[face.tetIdx[side]].x]);
      // PRINT(in.vertices[in.tets[face.tetIdx[side]].y]);
      // PRINT(in.vertices[in.tets[face.tetIdx[side]].z]);

      // PRINT(in.vertices[in.tets[tetIdx].x]);
      // PRINT(in.vertices[in.tets[tetIdx].y]);
      // PRINT(in.vertices[in.tets[tetIdx].z]);

      throw std::runtime_error("face with more than one tet on same side!?");
    }
    // std::cout <<" writing tet " << tetIdx << " / facet " << facetIdx << " to side " << side << " of face " << faceIdx << std::endl;
    face.tetIdx[side] = tetIdx;
    face.facetIdx[side] = facetIdx;

    out.tetFaces[tetIdx][facetIdx] = faceIdx;
  }
  
  /*! push tet #tetIdx from input mesh in, and fill in all respective
    values in the 'out' field */
  void TetConnHelper::pushTet(int tetIdx)
  {
    // int   sideOfFoundFace = /* will be filled in by 'findFace()1 */-1;
    vec4i index = in.tets[tetIdx];

    out.tetFaces.push_back(vec4i(-1));

    pushFacet(tetIdx,0,{index[1],index[3],index[2]});
    pushFacet(tetIdx,1,{index[0],index[2],index[3]});
    pushFacet(tetIdx,2,{index[0],index[3],index[1]});
    pushFacet(tetIdx,3,{index[0],index[1],index[2]});
  }
    

  /*! compute connectivity from given umesh; the original umesh will
    not be altered, and vertex and tet IDs in connectivity will
    refer to original umesh. Will throw an error for umeshes with
    any volume prims that are not tets */
  TetConn::SP TetConn::computeFrom(UMesh::SP umesh)
  {
    assert(umesh);
    TetConn::SP conn = std::make_shared<TetConn>();
    conn->computeFrom(*umesh);
    return conn;
  }
  
  /*! compute connectivity from given umesh; the original umesh will
    not be altered, and vertex and tet IDs in connectivity will
    refer to original umesh. Will throw an error for umeshes with
    any volume prims that are not tets */
  void TetConn::computeFrom(const UMesh &umesh)
  {
    TetConnHelper(*this,umesh);
  }
  
  /*! write - binary - to given file */
  void TetConn::write(std::ostream &out) const
  {
    io::writeVector(out,tetFaces);
    io::writeVector(out,faces);
  }
  
  /*! read from given file, assuming file format as used by saveTo() */
  void TetConn::read(std::istream &in)
  {
    io::readVector(in,tetFaces);
    io::readVector(in,faces);
  }

  /*! write - binary - to given file */
  void TetConn::saveTo(const std::string &fileName) const
  {
    std::ofstream out(fileName,std::ios::binary);
    write(out);
  }

  /*! read from given file, assuming file format as used by saveTo() */
  TetConn::SP TetConn::loadFrom(const std::string &fileName)
  {
    TetConn::SP conn = std::make_shared<TetConn>();
    std::ifstream in(fileName,std::ios::binary);
    conn->read(in);
    return conn;
  }
    
}
