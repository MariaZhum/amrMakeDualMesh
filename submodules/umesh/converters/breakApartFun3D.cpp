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

/* information on fund3d data format provided by Pat Moran - greatly
   indebted! All loader code rewritten from scratch. */

#include "umesh/io/ugrid32.h"
#include "umesh/io/ugrid64.h"
#include "umesh/io/fun3dScalars.h"
#include "umesh/RemeshHelper.h"

namespace umesh {

  std::string path = "";
  
  std::string scalarsPath = "";
  /*! list of ALL variables in first rank's scalars file */
  std::vector<std::string> variables;
  /*! list of ALL time steps in first rank's scalars file */
  std::vector<int>         timeSteps;
  
  bool doPart(const std::string &outFileNameBase, int rank)
  {
    std::cout << "----------- part " << rank << " -----------" << std::endl;
    std::string metaFileName = path + "ta."+std::to_string(rank);
    std::string meshFileName = path + "sh.lb4."+std::to_string(rank);
    std::cout << "reading from " << meshFileName << std::endl;
    std::cout << "     ... and " << metaFileName << std::endl;
    struct {
      int tets, pyrs, wedges, hexes;
    } meta;

    {
      FILE *metaFile = fopen(metaFileName.c_str(),"r");
      if (!metaFile) return false;
      assert(metaFile);
        
      int rc =
        fscanf(metaFile,
               "n_owned_tetrahedra %i\nn_owned_pyramids %i\nn_owned_prisms %i\nn_owned_hexahedra %i\n",
               &meta.tets,&meta.pyrs,&meta.wedges,&meta.hexes);
      assert(rc == 4);
      fclose(metaFile);
    }

    UMesh::SP mesh = io::UGrid32Loader::load(meshFileName);
    std::cout << "loaded part mesh " << mesh->toString() << " " << mesh->getBounds() << std::endl;
#if 1
    const std::string outFileNameMeshGhost = outFileNameBase + "." + std::to_string(rank) + "-with-ghost-cells.umesh";
    mesh->saveTo(outFileNameMeshGhost);

    mesh->tets.resize(meta.tets);
    mesh->pyrs.resize(meta.pyrs);
    mesh->wedges.resize(meta.wedges);
    mesh->hexes.resize(meta.hexes);
    // shrink to exclude the macrocells!
    std::cout << "after removing the ghost cells: " << mesh->toString() << std::endl;
#endif

    const std::string outFileNameMesh = outFileNameBase + "." + std::to_string(rank) + ".umesh";
    mesh->saveTo(outFileNameMesh);


    for (auto var : variables) {
      for (auto ts : timeSteps) {
          
        std::string scalarsFileName
          = scalarsPath
      //    + "volume_data."
          + std::to_string(rank);
        char ts_suffix[100];
        sprintf(ts_suffix,"__ts_%07i",ts);
        const std::string outFileNameScalars
          = outFileNameBase + "__var_" + var + ts_suffix + "." + std::to_string(rank) + ".floats";
        
        std::cout << "reading time step " << ts
                  << " from " << scalarsFileName << std::endl;
          
        std::vector<size_t> globalVertexIDs;
        /*! desired time step's scalars for current brick, if provided */
        std::vector<float> scalars
          = io::fun3d::readTimeStep(scalarsFileName,var,ts,
                                    &globalVertexIDs);
        for (int i=0;i<10;i++)
          std::cout << scalars[i] << " ";
        std::cout << std::endl;
        std::ofstream bin(outFileNameScalars,std::ios::binary);
        bin.write((const char *)scalars.data(),scalars.size()*sizeof(scalars[0]));
std::cout << UMESH_TERMINAL_GREEN 
<< " -> written to " << outFileNameScalars
<< UMESH_TERMINAL_DEFAULT << std::endl;
      }
    }
    std::cout << " >>> done part " << rank << ", got " << mesh->toString(false) << " (note it's OK that bounds aren't set yet)" << std::endl;
    return true;
  }
    
  void usage(const std::string &error = "")
  {
    if (error != "")
      std::cout << "Fatal error: " << error << std::endl << std::endl;
    std::cout << "./umeshBreakApartFun3D <path> <args>" << std::endl;
    std::cout << std::endl;
    std::cout << "takes a Fun3D model (eg, mars lander), and split it into its per-rank" << std::endl;
    std::cout << "components; writing one umesh per rank, and one matching 'raw' scalars" << std::endl;
    std::cout << "file per rank per variable per time step" << std::endl;
    std::cout << std::endl;
    std::cout << "w/ Args: " << std::endl;
    std::cout << "-o <outPath>\n\tbase part of filename for all output files" << std::endl;
    std::cout << "-n <numFiles> --first <firstFile>\n\t(optional) which range of files to process\n\te.g., --first 2 -n 3 will process files name.2, name.3, and name.4" << std::endl;
    std::cout << "--scalars scalarBasePath\n\twill read scalars from *_volume.X files at given <scalarBasePath>_volume.X" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "./umeshBreakApartFun3D /space/fun3d/small/dAgpu0145_Fa_ --scalars /space/fun3d/small/10000unsteadyiters/dAgpu0145_Fa_volume_data. -o /space/fun3d/merged_lander_small" << std::endl;
    std::cout << "to print available variables and time steps, call with file names but do not specify time step or variables" << std::endl;
    exit( error != "");
  }
  
  extern "C" int main(int ac, char **av)
  {
    int begin = 1;
    int num = 10000;

    std::string outFileBase = "";//"huge-lander.umesh";
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-n" || arg == "--num" || arg == "-num")
        num = atoi(av[++i]);
      else if (arg == "--first")
        begin = atoi(av[++i]);
      else if (arg == "-s" || arg == "--scalars")
        scalarsPath = av[++i];
      else if (arg == "-o")
        outFileBase = av[++i];
      else if (arg[0] != '-')
        path = arg;
      else
        usage("unknown cmdline arg "+arg);
    }
    if (path == "") usage("no input path specified");
    if (outFileBase == "") usage("no output filename specified");


    std::cout << "reading info on which times steps and fields there are ..." << std::endl;
    {
      std::string firstFileName = scalarsPath+std::to_string(begin);
      io::fun3d::getInfo(firstFileName,variables,timeSteps);
      std::cout << "File Info: " << std::endl;
      std::cout << "variables:";
      for (auto var : variables) std::cout << " " << var;
      std::cout << std::endl;
      std::cout << "timeSteps:";
      for (auto var : timeSteps) std::cout << " " << var;
      std::cout << std::endl;
    }
    
    std::cout << "OK, got the field info, now extracting ranks' data" << std::endl;
    for (int i=begin;i<(begin+num);i++)
      if (!doPart(outFileBase,i))
        break;

    std::cout << "done all ..." << std::endl;
  }
  
} // ::umesh
