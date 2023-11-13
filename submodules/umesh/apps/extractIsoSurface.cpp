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

#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/extractIsoSurface.h"

namespace umesh {

  void usage(const std::string error="")
  {
    if (error != "")
      std::cerr << "Error : " << error  << "\n\n";

    std::cout << "Usage: ./umeshExtractIsoSurface <in.umesh> -iso scalarValue (-o <out.umesh> | --obj file.obj)" << std::endl;;
    exit (error != "");
  };
  
  extern "C" int main(int ac, char **av)
  {
    float isoValue = std::numeric_limits<float>::infinity();
    std::string inFileName;
    std::string outFileName;
    std::string objFileName;
    /*! if enabled, we'll only save the tets that _we_ created, not
        those that were in the file initially */
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-h")
        usage();
      else if (arg == "-o")
        outFileName = av[++i];
      else if (arg == "-iso" || arg == "--iso-value" || arg == "--iso")
        isoValue = std::stof(av[++i]);
      else if (arg == "--obj")
        objFileName = av[++i];
      else if (arg[0] != '-')
        inFileName = arg;
      else
        usage("unknown cmd-line arg '"+arg+"'");
    }
    
    if (inFileName == "") usage("no input file specified");
    if (outFileName == "" && objFileName == "") usage("neither obj nor umesh output file specified");
    
    if (isoValue == std::numeric_limits<float>::infinity())
      usage("no iso-value specified");
    
    std::cout << "loading umesh from " << inFileName << std::endl;
    UMesh::SP in = UMesh::loadFrom(inFileName);
    
    std::cout << "done loading, found " << in->toString()
              << " ... now extracting iso-surface" << std::endl;
    if (in->pyrs.empty() &&
        in->wedges.empty() &&
        in->hexes.empty()) {
      std::cout << UMESH_TERMINAL_RED << std::endl;
      std::cout << "*******************************************************" << std::endl;
      std::cout << "WARNING: umesh already contains only tets..." << std::endl;
      std::cout << "*******************************************************" << std::endl;
      std::cout << UMESH_TERMINAL_DEFAULT << std::endl;
    }
    
    UMesh::SP result = extractIsoSurface(in,isoValue);
    std::cout << "done extracting isovalue, found " << result->toString() << std::endl;
    if (outFileName != "") {
      std::cout << "saving to " << outFileName << std::endl;
      result->saveTo(outFileName);
    }
    if (objFileName != "") {
      std::cout << "writing in OBJ format to " << objFileName << std::endl;
      std::cout << UMESH_TERMINAL_RED << "# WARNING - this can take a while!"
                << UMESH_TERMINAL_DEFAULT << std::endl;
      std::ofstream out(objFileName);
      for (auto v : result->vertices)
        out << "v " << v.x << " " << v.y << " " << v.z << std::endl;
      for (auto t : result->triangles)
        out << "f " << (t.x+1) << " " << (t.y+1) << " " << (t.z+1) << std::endl;
    }
      
    std::cout << "done all ..." << std::endl;
    
  }
} // ::umesh
