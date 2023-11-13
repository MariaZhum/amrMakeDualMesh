//Generation of special cases of cubes combinations 

#include "submodules/umesh/umesh/math.h"
#include <fstream>
#include <algorithm>
#include <random>

using namespace umesh;

int LEVEL = 0;
int MCWIDTH = 8;
vec3i WORLDSIZEINMC = vec3i(1, 1, 1);
bool SHUFFLE = false;
bool PRINTLOWER = false;


struct Cube{
  vec3f lower;
  int level;
  std::array<int, 8> scalarIDs;
};


//write Cubes for one level into file
void writeLevel(int level, const std::vector<Cube> &cubes, const std::string &outFileName){
    
  std::string fileName = outFileName+"_"+std::to_string(level)+".cubes"; 
  std::ofstream out(fileName,std::ios::binary);
  out.write((char*)cubes.data(), cubes.size()*sizeof(cubes[0]));
  std::cout << "done writting " << fileName << std::endl;

}

//prints lower coord of every cube
void printEveryLower(std::vector <Cube> cubesList, const std::string &lvlName){
  std::cout << "print cubes.lower for " << lvlName << "_" << cubesList[0].level << std::endl;
  for(auto cube: cubesList){
    std::cout << "(" << cube.lower.x << " " << cube.lower.y << " "<< cube.lower.z << ")" << std::endl;
  }
  std::cout << " " << std::endl;
}


std::vector <Cube> splitWholeLevel(int currentLevel, std::vector <Cube> &mainCubesVec){
  std::vector <Cube> cubesVec;
  if(currentLevel > 0){
    //int level = currentLevel/2;
    int level = currentLevel - 1;
    int newWidth = 1<<level;
    
    Cube tmpCube;
    for(auto rootCube:mainCubesVec){
      tmpCube.lower = rootCube.lower;
      tmpCube.level = level;
      tmpCube.scalarIDs = {1,1,1,1,1,1,1,1};
      cubesVec.push_back(tmpCube);
      //001 = (zyx)
      tmpCube.lower.x = tmpCube.lower.x + newWidth;
      cubesVec.push_back(tmpCube);
      //011
      tmpCube.lower.y = tmpCube.lower.y + newWidth;
      cubesVec.push_back(tmpCube);
      //111
      tmpCube.lower.z = tmpCube.lower.z + newWidth;
      cubesVec.push_back(tmpCube);
      //101
      tmpCube.lower.y = rootCube.lower.y;
      cubesVec.push_back(tmpCube);
      //100
      tmpCube.lower.x = rootCube.lower.x;
      cubesVec.push_back(tmpCube);
      //110
      tmpCube.lower.y = tmpCube.lower.y + newWidth;
      cubesVec.push_back(tmpCube);
      //010
      tmpCube.lower.z = rootCube.lower.z;
      cubesVec.push_back(tmpCube);
    }
    std::cout << cubesVec.size() << " cubes generated for lvl " << level << std::endl;
    writeLevel(level, cubesVec, "denseSplittedLevel");

    if(PRINTLOWER)
      printEveryLower(cubesVec, "denseSplittedLevel");
    
  }
  else{
    std::cout<< "It's already the smallest level!" << std::endl;
  }
  return cubesVec;
}


//gen huge cubes to be able to split them afterwards
std::vector<Cube> genCubeAsBigAsMC(int level = LEVEL){
  std::vector<Cube> cubesVec;
  int numOfCubes = WORLDSIZEINMC.x * WORLDSIZEINMC.y * WORLDSIZEINMC.z;
  cubesVec.resize(numOfCubes);

  for(int i=0; i<WORLDSIZEINMC.x; i++){
    for(int j=0; j<WORLDSIZEINMC.y; j++){
      for(int k=0; k<WORLDSIZEINMC.z; k++){
        cubesVec[k+j*WORLDSIZEINMC.z+i*WORLDSIZEINMC.z*WORLDSIZEINMC.y].lower = vec3f(i*1<<level, j*1<<level, k*1<<level);
        cubesVec[k+j*WORLDSIZEINMC.z+i*WORLDSIZEINMC.z*WORLDSIZEINMC.y].scalarIDs  = {1,1,1,1,1,1,1,1};
        cubesVec[k+j*WORLDSIZEINMC.z+i*WORLDSIZEINMC.z*WORLDSIZEINMC.y].level = level;
      }
    }
  }
  std::cout << cubesVec.size() << " cubes generated for lvl " << level << std::endl;
  writeLevel(level, cubesVec, "denseSplittedLevel");

  if(PRINTLOWER)
    printEveryLower(cubesVec, "denseSplittedLevel");

  return cubesVec;
}

void genDenseSplittedSet(int level = LEVEL){
  std::vector<Cube> mainVec = genCubeAsBigAsMC();
  int currLvl = level;
  while (currLvl>0){
    mainVec = splitWholeLevel(currLvl, mainVec);
    currLvl--;
  }
  
}

//one level of cubes with max density
void genDense(int level=LEVEL, vec3i worldSize = WORLDSIZEINMC, bool shuffle=SHUFFLE, int mcWidth=MCWIDTH){
  std::vector<Cube> cubesVec;
  vec3i absoluteWorldSize = vec3i(mcWidth*worldSize.x*1<<level, mcWidth*worldSize.y*1<<level, mcWidth*worldSize.z*1<<level);
  cubesVec.resize(worldSize.x*worldSize.y*worldSize.z* pow(MCWIDTH,3));
  for(int i = 0; i < cubesVec.size(); i++){
    cubesVec[i].level = level;
    cubesVec[i].scalarIDs  = {1,1,1,1,1,1,1,1};
    cubesVec[i].lower.x = i/(absoluteWorldSize.y*absoluteWorldSize.z);
    cubesVec[i].lower.y = (i/absoluteWorldSize.z)%absoluteWorldSize.y;
    cubesVec[i].lower.z = i%absoluteWorldSize.z;
  
  }

  std::cout << cubesVec.size() << " cubes generated for dense lvl " << level << std::endl;

  if(shuffle==true){
    std::shuffle(cubesVec.begin(), cubesVec.end(), std::random_device());
    std::cout << "shuffled cubes" << std::endl;
  }

  std::string outName = "denseLevel"+std::to_string(worldSize.x);

  writeLevel(level, cubesVec, outName);

  if(PRINTLOWER)
    printEveryLower(cubesVec, "denseLevel");
}

//one level of one Cube per brick
void genScarce(int level=LEVEL, vec3i worldsizeInMC=WORLDSIZEINMC, int mcWidth=MCWIDTH){
  std::vector<Cube> cubesVec;
  cubesVec.resize(worldsizeInMC.x*worldsizeInMC.y*worldsizeInMC.z);

  for(int i=0; i<worldsizeInMC.x; i++){
    for(int j=0; j<worldsizeInMC.y; j++){
      for(int k=0; k<worldsizeInMC.z; k++){
        cubesVec[k+j*worldsizeInMC.z+i*worldsizeInMC.z*worldsizeInMC.y].lower = vec3f(mcWidth*i*1<<level, mcWidth*j*1<<level, mcWidth*k*1<<level);
        cubesVec[k+j*worldsizeInMC.z+i*worldsizeInMC.z*worldsizeInMC.y].scalarIDs  = {1,1,1,1,1,1,1,1};
      }
    }
  }
  writeLevel(level, cubesVec, "scarceLevel");
  std::cout << cubesVec.size() << " cubes generated for scarce lvl " << level << std::endl;
  if(PRINTLOWER)
    printEveryLower(cubesVec, "scarceLevel");
}

void genDeep(int maxLevel=LEVEL, vec3f base = vec3f(0.0)){
  int currentLvl = maxLevel;
  Cube tmpCube;
  int cellWidth = 1<<currentLvl;

  while (currentLvl >= 0){
    std::vector<Cube> cubesVec;
    tmpCube.level = currentLvl;
    tmpCube.lower = base;
    //set 7 cubes in current lvl
    tmpCube.scalarIDs = {1,1,1,1,1,1,1,1};
    cubesVec.push_back(tmpCube);
    //001 = (zyx)
    tmpCube.lower.x = tmpCube.lower.x + cellWidth;
    cubesVec.push_back(tmpCube);
    //011
    tmpCube.lower.y = tmpCube.lower.y + cellWidth;
    cubesVec.push_back(tmpCube);
    //010
    tmpCube.lower.x = tmpCube.lower.x - cellWidth;
    cubesVec.push_back(tmpCube);
    //110
    tmpCube.lower.z = tmpCube.lower.z + cellWidth;
    cubesVec.push_back(tmpCube);
    //100
    tmpCube.lower.y = tmpCube.lower.y - cellWidth;
    cubesVec.push_back(tmpCube);
    //101
    tmpCube.lower.x = tmpCube.lower.x + cellWidth;
    cubesVec.push_back(tmpCube);
    
    //write current lvl
    writeLevel(currentLvl, cubesVec, "deepLevelSet");
    std::cout << cubesVec.size() << " cubes generated for deeplvl " << currentLvl << std::endl;

    if(PRINTLOWER)
      printEveryLower(cubesVec, "deepLevelSet");

    //set new base and lvl
    //111
    tmpCube.lower.y = tmpCube.lower.y + cellWidth;
    base = tmpCube.lower;
    currentLvl--;
    cellWidth = 1<<currentLvl;
  }
}

//one level of cubes with max density and given basis
vec3f genDenseWithBasis(vec3i numCubes, int level, vec3f basis){

  std::vector<Cube> cubesVec;
  cubesVec.resize(numCubes.x*numCubes.y*numCubes.z);

  int cubeWidth = 1<<level;

 int cubeidx = 0;
  for(int i=0; i<numCubes.x; i++){
    for(int j=0; j<numCubes.y; j++){
      for(int k= 0; k<numCubes.z; k++){
        cubesVec[cubeidx].level = level;
        cubesVec[cubeidx].scalarIDs  = {1,1,1,1,1,1,1,1};
        cubesVec[cubeidx].lower.x = (float)(i*cubeWidth)+basis.x;
        cubesVec[cubeidx].lower.y = (float)(j*cubeWidth)+basis.y;
        cubesVec[cubeidx].lower.z = (float)(k*cubeWidth)+basis.z;
        //std::cout << cubesVec[cubeidx].lower.x  << " " << cubesVec[cubeidx].lower.y << " " << cubesVec[cubeidx].lower.z <<std::endl;
        cubeidx++;
      }
      
    }
  }

  std::cout << cubesVec.size() << " cubes generated for denseWithBasis, level=" << level <<  " basis= " << basis.x <<
                                                  " "<< basis.y << " " <<basis.z <<std::endl;

  std::string outName = "denseLevel";

  writeLevel(level, cubesVec, outName);

  if(PRINTLOWER)
    printEveryLower(cubesVec, "denseLevel");

  return cubesVec.back().lower;
}


//20 dense Levels 
void denseLvls_20(){
  vec3f currentBasis = vec3f(0);
  //(1,1,1) vertex of last gen cube is the new basis
  vec3f lastCubelower = vec3f(0);
  int currentlvl;
  vec3i numCubes = vec3i(35*MCWIDTH);

  for(int i = 0; i < 20; i++){
    currentlvl = i;
    
    lastCubelower = genDenseWithBasis(numCubes, currentlvl, currentBasis);
    currentBasis.x = lastCubelower.x + (float)(1<<currentlvl);
    currentBasis.y = lastCubelower.y + (float)(1<<currentlvl);
    currentBasis.z = lastCubelower.z + (float)(1<<currentlvl);
  }

  printf("20 dense levels generated!");

}


int main(int argc, char *argv[]){
 

  bool done = false;
  int level;
  vec3i lvlSize;

  while (!done){
    std::cout << "-----------------------------------------------------" << std::endl;

    std::cout << "Choose level type: 1 = one scarce level, 2 = set of dense splitted levels, 3 = set of deep levels, 4 = one dense level, 5 = 20 dense levels with 280x280x280 cubes, 6 = done"<< std::endl;
    int type;
    std::cin >> type;

    switch (type){
    case 1:
      std::cout << "Enter the level you want to generate as an integer:" << std::endl;
      std::cin >> level;

      std::cout << "Enter the level size in macrocells you want to generate:" << std::endl;
      std::cin >> lvlSize.x >> lvlSize.y >> lvlSize.z; 

      genScarce(level, lvlSize);
      break;

    case 2:
      std::cout << "Enter the level you want to generate as an integer:" << std::endl;
      std::cin >> level;

      genDenseSplittedSet(level);
      break;
    
    case 3:
      std::cout << "Enter the level you want to generate as an integer:" << std::endl;
      std::cin >> level;

      genDeep(level);
      break;

    case 4:
      std::cout << "Enter the level you want to generate as an integer:" << std::endl;
      std::cin >> level;

      std::cout << "Enter the level size in macrocells you want to generate:" << std::endl;
      std::cin >> lvlSize.x >> lvlSize.y >> lvlSize.z; 

      genDense(level, lvlSize);
      break;

    case 5:
      denseLvls_20();
      break;

    case 6:
      done = true;
      break;
    
    default:
      std::cout << "no input" << std::endl;
      break;
    }
  }
  
}