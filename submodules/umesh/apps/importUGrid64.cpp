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

#include "umesh/io/ugrid64.h"
#include "umesh/io/UMesh.h"
#include "umesh/tetrahedralize.h"

namespace umesh {

  void usage(const std::string error="")
  {
    if (error != "")
      std::cerr << "Error : " << error  << "\n\n";

    std::cout << "Usage: ./umeshImportUGrid64 <in.ugrid64> <scalarsFile.bin> -o <out.umesh>" << std::endl;;
    exit (error != "");
  };

  UMesh::SP
  importOFF(const std::string &fileName)
  {
    UMesh::SP mesh = std::make_shared<UMesh>();
    mesh->perVertex = std::make_shared<Attribute>();

    FILE *file = fopen(fileName.c_str(),"r");
    // std::ifstream in(fileName);
    int numVerts, numTets;
    //in >> numVerts >> numTets;
    int rc = fscanf(file,"%i %i\n",&numVerts,&numTets);
    assert(rc == 2);
    for (int i=0; i<numVerts;i++) {
      vec3f v;
      double f;
      // in >> v.x >> v.y >> v.z >> f;
      int rc = fscanf(file,"%f %f %f %lf\n",&v.x,&v.y,&v.z,&f);
      assert(rc == 4);
      
      mesh->vertices.push_back(v);
      mesh->perVertex->values.push_back((float)f);
    }

    for (int i=0; i<numTets;i++) {
      vec4i tet;
      int rc = fscanf(file,"%i %i %i %i\n",&tet.x,&tet.y,&tet.z,&tet.w);
      assert(rc == 4);
      mesh->tets.push_back(tet); //-vec4i(1));
    }
    mesh->finalize();
    fclose(file);
    return mesh;
  }
  
  extern "C" int main(int ac, char **av)
  {
    std::string ugridFileName;
    std::string scalarsFileName;
    std::string outFileName;
    /*! if enabled, we'll only save the tets that _we_ created, not
        those that were in the file initially */
    bool skipActualTets = false;
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-h")
        usage();
      else if (arg == "-o")
        outFileName = av[++i];
      else if (arg[0] != '-') {
        if (ugridFileName == "")
          ugridFileName = arg;
        else if (scalarsFileName == "")
          scalarsFileName = arg;
        else
          usage("more than two file names specified!?");
      } else
        usage("unknown cmd-line arg '"+arg+"'");
    }
    
    if (ugridFileName == "") usage("no ugrid file specified");
    if (scalarsFileName == "") //usage("no scalars file specified");
      std::cout << "Warning: no scalars file specified!!!" << std::endl;
    if (outFileName == "") usage("no output file specified");
    
    std::cout << "loading off from " << ugridFileName << " + " << scalarsFileName << std::endl;
    UMesh::SP in = io::UGrid64Loader::load(ugridFileName,scalarsFileName);
    if (scalarsFileName == "")
      for (size_t i=0;i<in->vertices.size();i++)
        in->vertexTag.push_back(i);
    std::cout << "done loading, found " << in->toString() << std::endl;
    
    in->saveTo(outFileName);
    std::cout << "done ..." << std::endl;
  }
} // ::umesh
