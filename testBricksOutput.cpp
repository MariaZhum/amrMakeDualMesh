#include "submodules/umesh/umesh/math.h"
#include <cstring>
#include <set>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace umesh;

struct Brick {
    vec3i lower;
    int   level;
    vec3i numCubes;
    std::vector<int> scalarIDs;
};

//Morton codes
inline unsigned long long morton_encode3D(unsigned long long x, unsigned long long y, unsigned long long z)
{                                                                                            
    auto separate_bits = [](unsigned long long n)                                            
    {                                                                                        
        n &= 0b1111111111111111111111ull;                                                    
        n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
        n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
        n = (n ^ (n <<  8)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
        n = (n ^ (n <<  4)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
        n = (n ^ (n <<  2)) & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
        return n;                                                                            
    };                                                                                       
                                                                                             
    return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2);             
} 


struct is_smaller{
    bool operator()(const Brick &a, const Brick &b) const{
        return morton_encode3D(a.lower.x, a.lower.y, a.lower.z) < morton_encode3D(b.lower.x, b.lower.y, b.lower.z);
    }
};

//pairwise comparison of two grids files
void compareBricks(std::vector<Brick> &origBricks, std::vector<Brick> &compBricks){

    if(origBricks.size() != compBricks.size()){
        std::cout << "Size mismatch!" << std::endl;
        std::cout << "original number of Bricks: " << origBricks.size() << ", comp. number of Bricks: " << compBricks.size() << std::endl;
        return;
    }

    if(origBricks[0].level != compBricks[0].level){
        std::cout << "Level mismatch!" << std::endl;
        std::cout << "original level: " << origBricks[0].level << ", comp. level: " << compBricks[0].level<< std::endl;
        return;
    }

    for(int i = 0; i< origBricks.size(); i++){
        if(origBricks[i].lower != compBricks[i].lower){
            std::cout << "Bricks.lower mismatch for brick number "<< i << "!" << std::endl;
            std::cout << "original bricks.lower: " << origBricks[i].lower  << ", comp. bricks.lower: " << compBricks[i].lower << std::endl;
            return;
        }

        for(int j=0; j<8; j++){
            if(origBricks[i].scalarIDs[j]!=compBricks[i].scalarIDs[j]){
                std::cout << "Scalars mismatch!" << std::endl;
                std::cout << "occured for Brick " << i << " "<< origBricks[i].lower <<  std::endl;
                std::cout << "original:" << origBricks[i].scalarIDs[j] << ", comp.: " << compBricks[i].scalarIDs[j] << std::endl;
                std::cout << "complete: orig=" << origBricks[i].scalarIDs[0] << " " << origBricks[i].scalarIDs[1] << " " << origBricks[i].scalarIDs[2] <<" " << origBricks[i].scalarIDs[3] <<" " << origBricks[i].scalarIDs[4] <<" " << origBricks[i].scalarIDs[5] <<" " << origBricks[i].scalarIDs[6] <<" " << origBricks[i].scalarIDs[7] 
                                                << " comp=" << compBricks[i].scalarIDs[0] << " "<< compBricks[i].scalarIDs[1] << " " << compBricks[i].scalarIDs[2] << " "<< compBricks[i].scalarIDs[3] << " "<< compBricks[i].scalarIDs[4] << " "<< compBricks[i].scalarIDs[5] << " "<< compBricks[i].scalarIDs[6] << " "<< compBricks[i].scalarIDs[7] << " "<< std::endl;
                std::cout << compBricks[i-1].lower << " orig="<< origBricks[i-1].scalarIDs[0] << " " << origBricks[i-1].scalarIDs[1] << " " << origBricks[i-1].scalarIDs[2] <<" " << origBricks[i-1].scalarIDs[3] <<" " << origBricks[i-1].scalarIDs[4] <<" " << origBricks[i-1].scalarIDs[5] <<" " << origBricks[i-1].scalarIDs[6] <<" " << origBricks[i-1].scalarIDs[7] 
                                << " comp=" << compBricks[i-1].scalarIDs[0] << " "<< compBricks[i-1].scalarIDs[1] << " " << compBricks[i-1].scalarIDs[2] << " "<< compBricks[i-1].scalarIDs[3] << " "<< compBricks[i-1].scalarIDs[4] << " "<< compBricks[i-1].scalarIDs[5] << " "<< compBricks[i-1].scalarIDs[6] << " "<< compBricks[i-1].scalarIDs[7] << " "<< std::endl;
                return;
            }
        }

        if(origBricks[i].numCubes != compBricks[i].numCubes){
            std::cout << "Number of cubes mismatch!" << std::endl;
            std::cout << "Original number of cubes: " << origBricks[i].numCubes << ", compBricks[i].numCubes: " << compBricks[i].numCubes << std::endl;
            return;
        }
    }
    std::cout << "All tests completed. No mismatch found." << std::endl;
}



int main(int ac, char **av){
    std::vector <Brick> origBricks;
    std::vector <Brick> compBricks;

    std::cout << "first file - original, second file - to be compaired" << std::endl;

    //read first file
    std::ifstream in(av[1], std::ios::binary);

    while (!in.eof()) {
        Brick brick;
        in.read((char*)&brick.lower,sizeof(brick.lower));
        
        in.read((char*)&brick.level,sizeof(brick.level));
        in.read((char*)&brick.numCubes,sizeof(brick.numCubes));
        if (!in.good())
        break;
        brick.scalarIDs.resize((brick.numCubes.x+1)*(brick.numCubes.y+1)*size_t((brick.numCubes.z+1)));
        in.read((char*)brick.scalarIDs.data(),brick.scalarIDs.size()*sizeof(brick.scalarIDs[0])); 
        origBricks.push_back(brick);
    }

    //read second file
    std::ifstream in2(av[2],std::ios::binary);
    
    while (!in2.eof()) {
        Brick brick;
        in2.read((char*)&brick.lower,sizeof(brick.lower));
        in2.read((char*)&brick.level,sizeof(brick.level));
        in2.read((char*)&brick.numCubes,sizeof(brick.numCubes));
        if (!in2.good())
        break;
        brick.scalarIDs.resize((brick.numCubes.x+1)*(brick.numCubes.y+1)*size_t((brick.numCubes.z+1)));
        in2.read((char*)brick.scalarIDs.data(), brick.scalarIDs.size()*sizeof(brick.scalarIDs[0]));
        compBricks.push_back(brick);
    }

    //sort
    std::sort(origBricks.begin(), origBricks.end(), is_smaller());
    std::sort(compBricks.begin(), compBricks.end(), is_smaller());

    //pairwise comparison
    compareBricks(origBricks, compBricks);

    return 0;
}
