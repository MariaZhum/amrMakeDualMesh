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

#include "umesh/extractShellFaces.h"
#include "umesh/FaceConn.h"
#include "umesh/RemeshHelper.h"
# ifdef UMESH_HAVE_TBB
#  include "tbb/parallel_sort.h"
# endif
#include <set>
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

namespace umesh {

  // UMesh::SP dbg_input;

  /*! given a umesh with mixed volumetric elements, create a a new
      mesh of surface elemnts (ie, triangles and quads) that
      corresponds to the outside facing "shell" faces of the input
      elements (ie, all those that re not shared by two different
      elements. All surface elements in the output mesh will be
      INWARD facing. */
  UMesh::SP extractShellFaces(UMesh::SP mesh);
} // ::umesh

namespace umesh {
  
  /*! given a umesh with mixed volumetric elements, create a a new
      mesh of surface elemnts (ie, triangles and quads) that
      corresponds to the outside facing "shell" faces of the input
      elements (ie, all those that re not shared by two different
      elements. All surface elements in the output mesh will be
      OUTWARD facing. */
  UMesh::SP extractShellFaces(UMesh::SP input,
                              /*! if true, we'll create a new set of
                                vertices for ONLY the required
                                vertices. If false, we'll leave the
                                vertices array empty, and have the
                                vertex indices refer to the
                                original input mesh */
                              bool remeshVertices
                              )
  {
    FaceConn::SP faceConn = FaceConn::compute(input);
    auto &faces = faceConn->faces;

    assert(faces.empty() || !input->vertices.empty());
    UMesh::SP output = std::make_shared<UMesh>();

#if 1
    output->vertices = input->vertices;
    if (input->perVertex)  {
      output->perVertex = std::make_shared<Attribute>();
      output->perVertex->name = input->perVertex->name;
      output->perVertex->values = input->perVertex->values;
    }

    for (auto &face: faces) {
      if (face.vertexIdx.x < 0)
        // invalid face
        continue;
      
      if (face.onFront.primIdx < 0 && face.onBack.primIdx < 0) {
        PRINT(face.vertexIdx);
        throw std::runtime_error("face that has BOTH sides unused!?");
      } else if (face.onFront.primIdx < 0) {
        if (face.vertexIdx.w < 0) {
          // SWAP
          vec3i tri(face.vertexIdx.x,
                    face.vertexIdx.z,
                    face.vertexIdx.y);
          output->triangles.push_back(tri);
        } else {
          // SWAP
          vec4i quad(face.vertexIdx.x,
                     face.vertexIdx.w,
                     face.vertexIdx.z,
                     face.vertexIdx.y);
          output->quads.push_back(quad);
        }
      } else if (face.onBack.primIdx < 0) {
        if (face.vertexIdx.w < 0) {
          // NO SWAP
          vec3i tri(face.vertexIdx.x,
                    face.vertexIdx.y,
                    face.vertexIdx.z);
          output->triangles.push_back(tri);
        } else {
          // NO SWAP
          vec4i quad(face.vertexIdx.x,
                     face.vertexIdx.y,
                     face.vertexIdx.z,
                     face.vertexIdx.w);
          output->quads.push_back(quad);
        }
      } else {
        /* inner face ... ignore */
      }
    }
    if (remeshVertices)
      removeUnusedVertices(output);
#else
    
    
    RemeshHelper helper(*output);
    for (auto &face: faces) {
      if (face.vertexIdx.x < 0)
        // invalid face
        continue;
      
      if (face.onFront.primIdx < 0 && face.onBack.primIdx < 0) {
        PRINT(face.vertexIdx);
        throw std::runtime_error("face that has BOTH sides unused!?");
      } else if (face.onFront.primIdx < 0) {
        if (face.vertexIdx.w < 0) {
          // SWAP
          vec3i tri(face.vertexIdx.x,
                    face.vertexIdx.z,
                    face.vertexIdx.y);
          if (remeshVertices)
            helper.translate(&tri.x,3,input);
          output->triangles.push_back(tri);
        } else {
          // SWAP
          vec4i quad(face.vertexIdx.x,
                     face.vertexIdx.w,
                     face.vertexIdx.z,
                     face.vertexIdx.y);
          if (remeshVertices)
            helper.translate(&quad.x,4,input);
          output->quads.push_back(quad);
        }
      } else if (face.onBack.primIdx < 0) {
        if (face.vertexIdx.w < 0) {
          // NO SWAP
          vec3i tri(face.vertexIdx.x,
                    face.vertexIdx.y,
                    face.vertexIdx.z);
          if (remeshVertices)
            helper.translate(&tri.x,3,input);
          output->triangles.push_back(tri);
        } else {
          // NO SWAP
          vec4i quad(face.vertexIdx.x,
                     face.vertexIdx.y,
                     face.vertexIdx.z,
                     face.vertexIdx.w);
          if (remeshVertices)
            helper.translate(&quad.x,4,input);
          output->quads.push_back(quad);
        }
      } else {
        /* inner face ... ignore */
      }
    }
#endif
    // dbg_input = 0;
    return output;
  }
  
}
