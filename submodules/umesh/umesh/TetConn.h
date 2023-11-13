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

  struct TetConn {
    typedef std::shared_ptr<TetConn> SP;
    
    /*! write - binary - to given file */
    void saveTo(const std::string &fileName) const;
    
    /*! read from given file, assuming file format as used by saveTo() */
    static TetConn::SP loadFrom(const std::string &fileName);
    
    /*! write - binary - to given file */
    void write(std::ostream &out) const;
    
    /*! read from given file, assuming file format as used by saveTo() */
    void read(std::istream &in);

    /*! compute connectivity from given umesh; the original umesh will
        not be altered, and vertex and tet IDs in connectivity will
        refer to original umesh. Will throw an error for umeshes with
        any volume prims that are not tets */
    static TetConn::SP computeFrom(UMesh::SP umesh);

    /*! compute connectivity from given umesh; the original umesh will
        not be altered, and vertex and tet IDs in connectivity will
        refer to original umesh. Will throw an error for umeshes with
        any volume prims that are not tets */
    void computeFrom(const UMesh &umesh);

    struct Face {
      /*! vertex indices */
      vec3i   index;
      /*! tet on given side, or -1 */
      int     tetIdx[2] = { -1,-1 };
      /*! facetID of tet on that side, or -1 not N/A */
      uint8_t facetIdx[2] = { 255, 255 };
    };

    // ------------------------------------------------------------------
    // per-tet data
    // ------------------------------------------------------------------
    
    /*! gives the four face IDs (indexing into the 'perFace[]'
      vectors) for a given tet. For each tet, facet #i is the face
      opposite vertex #i */
    std::vector<vec4i> tetFaces;

    // ------------------------------------------------------------------
    // per-face data
    // ------------------------------------------------------------------
    
    std::vector<Face> faces;
  };
  
} // :: umesh
