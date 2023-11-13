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

#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/check.h"

namespace umesh {

  void usage(const std::string error="")
  {
    if (error != "")
      std::cerr << "\nError : " << error  << "\n\n";

    std::cout << "Usage: ./umeshSanityCheck <in.umesh>\n\n";
    exit(error != "");
  };
  
  extern "C" int main(int ac, char **av)
  {
    std::string inFileName;
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-h")
        usage();
      else if (arg[0] != '-')
        inFileName = arg;
      else
        usage("unknown cmd-line arg '"+arg+"'");
    }
    
    if (inFileName == "") usage("no input file specified");
    
    std::cout << "loading umesh from " << inFileName << std::endl;
    UMesh::SP in = io::loadBinaryUMesh(inFileName);

    std::cout << "UMesh info:\n" << in->toString(false) << std::endl;
    sanityCheck(in);
    std::cout << "all sanity checks went through ..." << std::endl;
  }
  
} // ::umesh
