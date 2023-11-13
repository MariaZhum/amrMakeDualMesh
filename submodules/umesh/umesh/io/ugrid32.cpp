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

#include "ugrid32.h"
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
  namespace io {
    
    UMesh::SP UGrid32Loader::load(const std::string &dataFileName,
                                  const std::string &scalarFileName)
    {
      return UGrid32Loader(dataFileName,scalarFileName).result;
    }

    inline bool notDegenerate(const std::vector<vec3f> &vertices,
                              const uint32_t index[],
                              const size_t N)
    {
      box3f bounds;
      for (int i=0;i<N;i++) {
        if (int(index[i]) < 0) throw std::runtime_error("negative index!?");
        assert(int(index[i]) >= 0);
        assert(index[i] < vertices.size());
        bounds.extend(vertices[index[i]]);
      }
      bool degen// = (area(bounds) == 0.f);
        =  (bounds.lower.x==bounds.upper.x)
        || (bounds.lower.y==bounds.upper.y)
        || (bounds.lower.z==bounds.upper.z);
      // if (degen) PRINT(bounds);
      if (N == 4 &&
          (vertices[index[0]] == vertices[index[1]] ||
           vertices[index[0]] == vertices[index[2]] ||
           vertices[index[0]] == vertices[index[3]] ||
           vertices[index[1]] == vertices[index[2]] ||
           vertices[index[1]] == vertices[index[3]] ||
           vertices[index[2]] == vertices[index[3]])) {
        if (!degen) {
          static int extraDegenTets = 0;
          // PRINT(bounds);
          // PRINT(index);
          // PRINT(vertices[index[0]]);
          // PRINT(vertices[index[1]]);
          // PRINT(vertices[index[2]]);
          // PRINT(vertices[index[3]]);
          extraDegenTets++;
          // PRINT(extraDegenTets);
          // for (int i=0;i<4;i++) {
          //   PRINT(index[i]); PRINT(vertices[index[i]]);
          // }
        }
        degen = true;
      }

      static size_t numTests = 0;
      ++ numTests;
      
      if (degen) {

        // PING;
        // for (int i=0;i<N;i++) {
        //   PRINT(index[i]);
        //   PRINT(vertices[index[i]]);
        // }
        
        static size_t numDegen = 0;
        static size_t nextPing = 1;
        if (++numDegen >= nextPing) {
      if (verbose)
          std::cout << "num degen : " << numDegen << " / " << numTests << std::endl;
          // PRINT(numDegen);
          // PRINT(nu
          nextPing *= 2;
        }
      }
      return !degen;
    }
    



    


    UGrid32Loader::UGrid32Loader(const std::string &dataFileName,
                                 const std::string &scalarFileName)
    {
      if (verbose)
        std::cout << "#tetty.io: reading ugrid32 file ..." << std::endl;
      result = std::make_shared<UMesh>();

      std::ifstream data(dataFileName, std::ios_base::binary);

      struct {
        uint32_t n_verts, n_tris, n_quads, n_tets, n_pyrs, n_prisms, n_hexes;
      } header;
      
      readElement(data,header);
      size_t numDegen = 0;
      
      result->bounds = box3f(); 
      if (verbose)
     std::cout << "#tetty.io: reading " << prettyNumber(header.n_verts) << " vertices ..." << std::endl;
      result->vertices.reserve(header.n_verts);
      float pos[3];
      for (size_t i=0;i<header.n_verts;i++) {
        readArray(data,pos,3);
        const vec3f v(vec3f(pos[0],pos[1],pos[2]));

        if (pos[0] < -1e20f ||
            pos[1] < -1e20f ||
            pos[2] < -1e20f ||
            pos[0] > +1e20f ||
            pos[1] > +1e20f ||
            pos[2] > +1e20f) {
      if (verbose)
          std::cout << "Degen vertex " << i << " " << pos << std::endl;
        }
        result->vertices.push_back(v);
      }

      if (scalarFileName != "") {
        std::ifstream scalar(scalarFileName, std::ios::binary);
      if (verbose)
        std::cout << "#tetty.io: reading " << prettyNumber(header.n_verts) << " scalars ..." << std::endl;
        result->perVertex = std::make_shared<Attribute>();
        for (size_t i=0;i<header.n_verts;i++) {
          // double val;
          float val;
          readElement(scalar,val);
          result->perVertex->values.push_back(val);
          if (val < -1e20f ||
              val > +1e20f) {
      if (verbose)
            std::cout << "Degen vertex " << i << " " << val << std::endl;
          }
        }

        result->perVertex->finalize();
      }
      
      // tris
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_tris) << " triangles ..." << std::endl;
      result->triangles.reserve(header.n_tris);
      for (size_t i=0;i<header.n_tris;i++) {
        uint32_t idx[3];
        readArray(data,idx,3);
        for (int i=0;i<3;i++) idx[i] -= 1;

        if (notDegenerate(result->vertices,idx,3))
          result->triangles.push_back(vec3i((int)idx[0], (int)idx[1], (int)idx[2]));
      }

      // quads
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_quads) << " quads ..." << std::endl;
      result->quads.reserve(header.n_quads);
      for (size_t i=0;i<header.n_quads;i++) {
        uint32_t idx[4];
        readArray(data,idx,4);
        for (int i=0;i<4;i++) idx[i] -= 1;
        
        if (notDegenerate(result->vertices,idx,4))
          result->quads.push_back(vec4i((int)idx[0], (int)idx[1], (int)idx[2], (int)idx[3]));
      }

      if (verbose)
      std::cout << "#tetty.io: skipping " << (header.n_tris+header.n_quads) << " surface IDs" << std::endl;
      std::vector<uint32_t> surfaceIDs(header.n_tris+header.n_quads);
      readArray(data,surfaceIDs.data(),surfaceIDs.size());
      surfaceIDs.clear();

      // tets
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_tets) << " tets ..." << std::endl;
      result->tets.reserve(header.n_tets);
      
      std::vector<uint32_t> vecIndices(4*header.n_tets);
      readArray(data,vecIndices.data(),vecIndices.size());
      std::vector<UMesh::Tet> vecTets(header.n_tets);
      std::vector<bool> isGood(vecTets.size());
      serial_for_blocked
        (/*range:*/0,vecTets.size(),/*partsize:*/1024*1024,
         [&](const size_t begin, const size_t end){
           for (size_t i=begin;i<end;i++) {
             uint32_t idx[4] = {
                (uint32_t)vecIndices[4*i+0]-1,
                (uint32_t)vecIndices[4*i+1]-1,
                (uint32_t)vecIndices[4*i+2]-1,
                (uint32_t)vecIndices[4*i+3]-1
             };
             const UMesh::Tet tet((int)idx[0],
                                  (int)idx[1],
                                  (int)idx[2],
                                  (int)idx[3]);

             isGood[i] = notDegenerate(result->vertices,idx,4);
             vecTets[i] = tet;
           }});

      for (size_t i=0;i<header.n_tets;i++)
        if (isGood[i])
          result->tets.push_back(vecTets[i]);
      
      // pyrs
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_pyrs) << " pyramids ..." << std::endl;
      result->pyrs.reserve(header.n_pyrs);
      for (size_t i=0;i<header.n_pyrs;i++) {
        uint32_t idx[5];
        readArray(data,idx,5);
        for (int i=0;i<5;i++) idx[i] -= 1;

        if (notDegenerate(result->vertices,idx,5))
          result->pyrs.push_back
            ({(int)idx[0],(int)idx[1],(int)idx[2],(int)idx[3],(int)idx[4]});
      }
      
      // prims 
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_prisms) << " prisms ..." << std::endl;
      result->wedges.reserve(header.n_prisms);
      for (size_t i=0;i<header.n_prisms;i++) {
        uint32_t idx[6];
        readArray(data,idx,6);
        for (int i=0;i<6;i++) idx[i] -= 1;

        if (notDegenerate(result->vertices,idx,6))
          result->wedges.push_back
#if 1
            /*! APPARENTLY, ugrid32 dot NOT use the VTK ordering for
                wedges, but has front and back side swapped out */
            ({(int)idx[3],(int)idx[4],(int)idx[5],
              (int)idx[0],(int)idx[1],(int)idx[2]
              });
#else
            ({(int)idx[0],(int)idx[1],(int)idx[2],
              (int)idx[3],(int)idx[4],(int)idx[5]});
#endif
      }
      
      // hexes - TODO
      if (verbose)
      std::cout << "#tetty.io: reading " << prettyNumber(header.n_hexes) << " hexes ..." << std::endl;
      result->hexes.reserve(header.n_hexes);
      for (size_t i=0;i<header.n_hexes;i++) {
        uint32_t idx[8];
        readArray(data,idx,8);
        for (int i=0;i<8;i++) idx[i] -= 1;

        if (notDegenerate(result->vertices,idx,8))
          result->hexes.push_back
            ({ (int)idx[0],(int)idx[1],(int)idx[2],(int)idx[3],
               (int)idx[4],(int)idx[5],(int)idx[6],(int)idx[7]});
      }

      if (verbose)
      std::cout << "#tetty.io: done reading ...." << std::endl;

      result->finalize();
    }
    
  } // ::tetty::io
} // ::tetty
