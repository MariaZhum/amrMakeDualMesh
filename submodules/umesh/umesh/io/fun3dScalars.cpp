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

#include "umesh/io/fun3dScalars.h"

namespace umesh {
  namespace io {
    namespace fun3d {
      
      /*! class that loads a fun3d "volume_data" file */
      struct Fun3DScalarsReader {

        Fun3DScalarsReader(const std::string &fileName)
        {
          in.open(fileName,std::ios::binary);
          uint32_t magicNumber = io::readElement<uint32_t>(in);

          std::string versionString = io::readString(in);
     //     std::cout << "Found fun3d file with version string "
      //              << versionString << std::endl;
      
          uint32_t ignore = io::readElement<uint32_t>(in);
          numScalars = io::readElement<uint32_t>(in);
          
          variableNames.resize(io::readElement<uint32_t>(in));
          for (auto &var : variableNames) {
            var = io::readString(in);
       //     std::cout << "Found variable '" << var << "'" << std::endl;
          }

          globalVertexIDs.resize(numScalars);
          io::readArray(in,globalVertexIDs.data(),globalVertexIDs.size());
          // std::cout << "found num vertices " << globalVertexIDs.size() << std::endl;
          // std::cout << "found num vars " << variableNames.size() << std::endl;

          size_t dataBegin = in.tellg();
          for (int tsNo=0;;tsNo++) {
            try {
              uint32_t timeStepID = io::readElement<uint32_t>(in);
              timeStepOffsets[timeStepID] = in.tellg();
              // std::cout << "time step " << timeStepID << " at ofs " << timeStepOffsets[timeStepID] << std::endl;
              in.seekg(dataBegin
                       +tsNo*(variableNames.size()*globalVertexIDs.size()*sizeof(float)
                              +sizeof(timeStepID)),
                       std::ios::beg);
            } catch (std::exception e) {
              in.clear(in.goodbit);
              break;
            };
          }
        }

        void readTimeStep(std::vector<float> &scalars,
                          const std::string &desiredVariable,
                          int desiredTimeStep)
        {
          scalars.resize(globalVertexIDs.size());
      
          /* offsets based on _blocks_ of time steps (one per variable) */
          auto it = timeStepOffsets.find(desiredTimeStep);
          if (it == timeStepOffsets.end())
            throw std::runtime_error("could not find requested time step #"+std::to_string(desiredTimeStep)+"!");
          size_t offset = it->second;

          /* offsets based on which variable */
          int varID = 0;
          for (varID=0;;varID++) {
            if (varID >= variableNames.size())
              throw std::runtime_error("couldn't find requested variable");
            if (variableNames[varID] == desiredVariable)
              break;
          }
          std::vector<float> timeStepForAllVariables(scalars.size()*variableNames.size());
          std::cout << "seeking " << offset << std::endl;
          in.seekg(offset,std::ios::beg);
          io::readArray(in,timeStepForAllVariables.data(),timeStepForAllVariables.size());
          std::cout << "read array " << std::endl;
          // for (int i=0;i<20;i++) std::cout << " " << timeStepForAllVariables[i];
          // std::cout << std::endl;
          
          for (int i=0;i<scalars.size();i++) {
            scalars[i] = timeStepForAllVariables[varID+i*variableNames.size()];
            // if (i < 20) std::cout << " / " << scalars[i];
            // if (scalars[i] > 1e10f)
            //   std::cout << "BIG " << i << " = " << scalars[i] << std::endl;
          }
          // std::cout <<std::endl;
        }

        size_t numScalars;
        std::ifstream in;
        std::vector<std::string> variableNames;
        std::vector<size_t>      globalVertexIDs;
        /*! .first is time step ID, .second is the offset in the file */
        std::map<int,size_t>  timeStepOffsets;
        size_t sizeOfTimeStep;
      };
    
      /*! read header from fun3d data file, and reutrn info on what's
        contained */
      void getInfo(const std::string &scalarsFileName,
                   std::vector<std::string> &variables,
                   std::vector<int> &timeSteps)
      {
        Fun3DScalarsReader scalarReader(scalarsFileName);
        variables = scalarReader.variableNames;
        timeSteps.clear();
        for (auto it : scalarReader.timeStepOffsets)
          timeSteps.push_back(it.first);
      }
  
      /*! read one time step for one variable, from given file */
      std::vector<float> readTimeStep(const std::string &scalarsFileName,
                                      const std::string &desiredVariable,
                                      int desiredTimeStep,
                                      std::vector<uint64_t> *globalIDs)
      {
        Fun3DScalarsReader scalarReader(scalarsFileName);
        std::vector<float> result;
        scalarReader.readTimeStep(result,
                                  desiredVariable,
                                  desiredTimeStep);
        if (globalIDs)
          *globalIDs = scalarReader.globalVertexIDs;
        return result;
      }

    }
  }
}
