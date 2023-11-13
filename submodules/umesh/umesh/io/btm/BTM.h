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

#include "umesh/UMesh.h"
#include "umesh/io/IO.h"

namespace umesh {
  /*! a binary-triangle mesh format, where each file contains exactly
    one mesh made up of vertices, vertex normals, vertex colors,
    texcoords, vertex indices, and a per-triangle color. all vectors
    other than 'vertex' and 'index' fields may be empty (but _will_
    be written as zero-length vectors) */
  namespace btm {
    struct Mesh {
      typedef std::shared_ptr<Mesh> SP;

      static Mesh::SP load(const std::string &fileName);
        
      void save(const std::string &fileName) const;

      /*! load tihs mesh's vectors from current positoin in given
        (binary) input stream) */
      void loadFrom(std::ifstream &in);
      void saveTo(std::ofstream &out) const;

      /*! vertex positoins, may not be empty */
      std::vector<vec3f> vertex;
        
      /*! per-vertex normal; may be empty, but if not empty must be
        same size as 'vertex' */
      std::vector<vec3f> normal;
        
      /*! per-vertex color; may be empty, but if not empty must be
        same size as 'vertex' */
      std::vector<vec3f> color;

      /*! per-vertex texture coordinates; may be empty, but if not
        empty must be same size as 'vertex' */
      std::vector<vec2f> texcoord;

      /*! per-triangle vertex indices; starting indexing at 0 */
      std::vector<vec3i> index;
        
      /*! per-triangle color value; may be empty, or same size as
        'index' */
      std::vector<vec3f> triColor;
    };
      
  } // ::umesh::btm
} // ::umesh
