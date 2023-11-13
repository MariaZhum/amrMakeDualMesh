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


#pragma once

#include "umesh/UMesh.h"

namespace umesh {

  /*! structure that encodes the full face-connectivy of a input mesh
      made up of unstructured elemens (only considering the volume
      elements of the input). This can be useful to compute, for
      example, the boundary face, or to allow marching or sampling
      algorithms that require nowing what's on a face, or on the the
      respectivly other sdie of a face, in a given mesh. Computing
      this is only well defined for "well defined" meshes, and amy
      fail or "bad" meshes with more than two prims sharing the same
      face, etc. */
  struct FaceConn {
    /*! allows to reference a "facet" of an input primitive - ie, the
        six'th face of prim number P would be encoded as "facetIdx=6",
        "primIdx=P", "primType=HEX". Note we intentionally use the
        term "facet" here instead of "face", to avoid the ambiguities
        that we get otherwise when talking about faces of the mesh,
        and faces of a specifci prim - those are two very different
        things - a facet is unique to a specific prim; but a face can
        be shared between two differnet prims */
    struct PrimFacetRef{
      /*! only 4 possible types (tet, pyr, wedge, or hex) */
      uint64_t primType:3;
      /*! only 8 possible facet it could be (in a hex) */
      uint64_t facetIdx:3;
      int64_t  primIdx:58;
    };

    /*! a shared face in the mode; given by four vertex indices (to
        represent either tri or quad), and two 'PrimFacetRef's to
        reference the type and ID of prim on front and back side of
        each face, respectively */
    struct SharedFace {
      /*! the four vertex indices (pointing into the input mesh) that
        make this face; if the face is a triangle, vtxIdx.w will be
        -1) */
      vec4i vertexIdx;
    
      /*! the input mesh's priitive on the front resp back side of tihs
        face; for boundary faces one of those may be tagged as
        invalid, otherwise they refer to prims in the input mesh */
      PrimFacetRef onFront, onBack;
    };


    typedef std::shared_ptr<FaceConn> SP;

    /*! given a unstructured mesh, compute the face-connectivity for
        this mesh. Note this _sohuld_ work even for curved/bilinear
        faces, but will error out for meshes with bad connectivyt
        (faces with more than two owning prims) */
    static FaceConn::SP compute(UMesh::SP mesh);

    /*! write - binary - to given file */
    void saveTo(const std::string &fileName) const;
    
    /*! read from given file, assuming file format as used by saveTo() */
    static FaceConn::SP loadFrom(const std::string &fileName);
    
    /*! write - binary - to given file */
    void write(std::ostream &out) const;
    
    /*! read from given file, assuming file format as used by saveTo() */
    void read(std::istream &in);

    
    // /*! the mesh that this face-connectivty belongs to */
    // UMesh::SP mesh;
    
    /*! the list of all faces in the given umesh; each tagged with the
        prim on the front resp back side of that face. each face can
        be either a tri (in which case the idx.w for the vertex
        indices will be -1), or a quad */
    std::vector<SharedFace> faces;
  };

  std::ostream &operator<<(std::ostream &out, const FaceConn::PrimFacetRef &ref);
} // ::umesh
