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

/* no computations at all - just dumps the mesh that's implicit in the
   input umesh, and dumps it in obj format (other prims get
   ignored) */ 

#include "umesh/io/ugrid64.h"
#include "umesh/io/UMesh.h"
#include "umesh/RemeshHelper.h"
#include "umesh/extractShellFaces.h"
#include <algorithm>

namespace umesh {

  typedef enum { INVALID, UMESH, OBJ } Format;

  Format formatFromFileName(const std::string &fileName)
  {
    if (fileName.substr(fileName.size()-4) == ".obj") return OBJ;
    if (fileName.substr(fileName.size()-6) == ".umesh") return UMESH;
    return INVALID;
  }

  inline bool flat(const vec3f &a,
                   const vec3f &b,
                   const vec3f &c,
                   const vec3f &d)
  {
    vec3f n0 = cross(b-a,c-a);
    if (dot(n0,n0) == 0.f) return false;
    vec3f n1 = cross(c-a,d-a);
    if (dot(n1,n1) == 0.f) return false;

    n0 = normalize(n0);
    n1 = normalize(n1);
    return dot(n0,n1) >= .99f;
  }
  void saveToOBJ(const std::string &outFileName, UMesh::SP mesh)
  {
    std::cout << "... saving (in OBJ format) to " << outFileName << std::endl;
    std::ofstream out(outFileName);
    for (auto vtx : mesh->vertices)
      out << "v " << vtx.x << " " << vtx.y << " " << vtx.z << std::endl;
    for (auto idx : mesh->triangles)
      out << "f " << (idx.x+1) << " " << (idx.y+1) << " " << (idx.z+1) << std::endl;
    for (auto idx : mesh->quads) {
      vec3f v0 = mesh->vertices[idx.x];
      vec3f v1 = mesh->vertices[idx.y];
      vec3f v2 = mesh->vertices[idx.z];
      vec3f v3 = mesh->vertices[idx.w];
      if (flat(v0,v1,v2,v3)) {
        out << "f " << (idx.x+1) << " " << (idx.y+1) << " " << (idx.z+1) << " " << (idx.w+1) << std::endl;
      } else {
        // write mini-bilinear patch of NxN vertices ((N-1)x(N-1) quads)
        int N = 6;
        for (int ix=0;ix<N;ix++)
          for (int iy=0;iy<N;iy++) {
            float u = ix / float(N-1);
            float v = iy / float(N-1);
            vec3f p
              = (1.f-u)*(1.f-v)*v0
              + (1.f-u)*(    v)*v1
              + (    u)*(1.f-v)*v3
              + (    u)*(    v)*v2
              ;
            out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
          }
        for (int ix=0;ix<N-1;ix++)
          for (int iy=0;iy<N-1;iy++) {
            out << "f " << ((ix+0)*N+(iy+0)-N*N)
                << "  " << ((ix+1)*N+(iy+0)-N*N)
                << "  " << ((ix+1)*N+(iy+1)-N*N)
                << "  " << ((ix+0)*N+(iy+1)-N*N) << std::endl;
          }
      }
    }
    std::cout << "... done" << std::endl;
  }

  UMesh::SP load(const std::string &fileName)
  {
    if (fileName.substr(fileName.size()-6) == ".umesh")
      return UMesh::loadFrom(fileName);
    
    if (fileName.substr(fileName.size()-8) == ".ugrid64")
      return io::UGrid64Loader::load(fileName);

    throw std::runtime_error("could not determine input format"
                             " (only supporting ugrid64 or umesh for now)");
  }
  
  extern "C" int main(int ac, char **av)
  {
    try {
      std::string inFileName;
      std::string outFileName;
      Format format = INVALID;

      for (int i = 1; i < ac; i++) {
        const std::string arg = av[i];
        if (arg == "-o")
          outFileName = av[++i];
        else if (arg == "--obj")
          format = OBJ;
        else if (arg == "--umesh")
          format = UMESH;
        else if (arg[0] != '-')
          inFileName = arg;
        else {
          throw std::runtime_error("./umeshExtractShell <in.umesh> [--obj|--umesh] -o <out.obj|.umesh>");
        }
      }

      if (format == INVALID)
        format = formatFromFileName(outFileName);
      
      std::cout << "loading umesh from " << inFileName << std::endl;
      UMesh::SP inMesh = load(inFileName);

      std::cout << "extracting shell faces .... this can take a while" << std::endl;
      UMesh::SP outMesh = extractShellFaces(inMesh,1);

      std::cout << "extracted surface of " << outMesh->toString() << std::endl;
      switch (format) {
      case OBJ:
        saveToOBJ(outFileName,outMesh);
        break;
      case UMESH:
        outMesh->saveTo(outFileName);
        break;
      default:
        throw std::runtime_error("invalid/unsupported format!?");
      }
    }
    catch (std::exception &e) {
      std::cerr << "fatal error " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }  
} // ::umesh
