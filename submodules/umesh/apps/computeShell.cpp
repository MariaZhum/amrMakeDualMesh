DO NOT USE ANY MORE - USE extractShellFaces instead
// // ======================================================================== //
// // Copyright 2018-2020 Ingo Wald                                            //
// //                                                                          //
// // Licensed under the Apache License, Version 2.0 (the "License");          //
// // you may not use this file except in compliance with the License.         //
// // You may obtain a copy of the License at                                  //
// //                                                                          //
// //     http://www.apache.org/licenses/LICENSE-2.0                           //
// //                                                                          //
// // Unless required by applicable law or agreed to in writing, software      //
// // distributed under the License is distributed on an "AS IS" BASIS,        //
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// // See the License for the specific language governing permissions and      //
// // limitations under the License.                                           //
// // ======================================================================== //

// /* computes the outer shell of a tet-mesh, ie, all the triangle and/or
//    bilinear faces that are _not_ shared between two neighboring
//    elements */ 

// #include "umesh/io/ugrid32.h"
// #include "umesh/io/UMesh.h"
// #include "umesh/io/btm/BTM.h"
// #include "umesh/RemeshHelper.h"
// #include <algorithm>

// namespace umesh {

//   struct ShellHelper
//   {
//     typedef std::vector<int> FaceKey;
    
//     struct Face {
//       typedef std::shared_ptr<Face> SP;
//       std::vector<std::pair<UMesh::PrimRef,int>> primsAndLocalFace;
      
//       /*! stores, for each vertex in this face, the list of indices AS
//         THEY REFER TO THE ORIGINAL (VOLUMETRIC!) MESH's vertex
//         array */
//       std::vector<int> originalIndices;

//       /*! stores, for each vertex in this face, the list of indices AS
//         THEY REFER TO THE (SURFACE-ONLY, REMESHES) 'out' mesh */
//       std::vector<int> remeshedIndices;
//     };
    
//     ShellHelper(UMesh::SP in)
//       : in(in),
//         out(std::make_shared<UMesh>()),
//         indexer(*out)
//     {
//       addFaces();
//     }

//     FaceKey computeKey(std::vector<int> faceIndices)
//     {
//       std::sort(faceIndices.begin(),faceIndices.end());
//       return faceIndices;
//     }

//     Face::SP findFace(const std::vector<int> &faceIndices)
//     {
//       FaceKey key = computeKey(faceIndices);
//       auto it = faces.find(key);
//       if (it != faces.end()) return it->second;

//       Face::SP face = std::make_shared<Face>();
//       face->originalIndices = faceIndices;

//       for (auto org : faceIndices)
//         face->remeshedIndices.push_back(indexer.getID(in->vertices[org],
//                                                       (size_t)org));
//       faces[key] = face;

//       auto &idx = face->remeshedIndices;
//       if (idx.size() == 3)
//         out->triangles.push_back({idx[0],idx[1],idx[2]});
//       else
//         out->quads.push_back({idx[0],idx[1],idx[2],idx[3]});
      
//       return face;
//     }

//     void addFace(const std::vector<int> &faceIndices,
//                  UMesh::PrimRef primRef,
//                  int localFaceID)
//     { findFace(faceIndices)->primsAndLocalFace.push_back({primRef,localFaceID}); }

//     void addFaces(const Tet &tet, UMesh::PrimRef primRef)
//     {
//       addFace({tet[0],tet[1],tet[2]},primRef,3);
//       addFace({tet[0],tet[1],tet[3]},primRef,2);
//       addFace({tet[0],tet[2],tet[3]},primRef,1);
//       addFace({tet[1],tet[2],tet[3]},primRef,0);
//     }
    
//     void addFaces(const Pyr &pyr, UMesh::PrimRef primRef)
//     {
//       {
//         addFace({pyr[0],pyr[4],pyr[3]},primRef,0);
//         addFace({pyr[1],pyr[2],pyr[4]},primRef,1);
//         addFace({pyr[0],pyr[1],pyr[4]},primRef,2);
//         addFace({pyr[4],pyr[3],pyr[2]},primRef,3);

//         addFace({pyr[0],pyr[3],pyr[2],pyr[1]},primRef,4);
//       }
//     }
    
//     void addFaces(const Wedge &wedge, UMesh::PrimRef primRef)
//     {
//       addFace({wedge[0],wedge[3],wedge[5],wedge[2]},primRef,0);
//       addFace({wedge[1],wedge[4],wedge[5],wedge[2]},primRef,1);
      
//       addFace({wedge[0],wedge[1],wedge[2]},primRef,2);
//       addFace({wedge[3],wedge[5],wedge[4]},primRef,3);

//       addFace({wedge[0],wedge[1],wedge[4],wedge[3]},primRef,4);
//     }
//     void addFaces(const Hex &hex, UMesh::PrimRef primRef)
//     {
//       // windinroder.png: x dir, left to right
//       addFace({hex[0],hex[3],hex[7],hex[4]},primRef,0);
//       addFace({hex[1],hex[2],hex[6],hex[5]},primRef,1);

//       // y dir, front to back
//       addFace({hex[0],hex[1],hex[5],hex[4]},primRef,2);
//       addFace({hex[2],hex[6],hex[7],hex[3]},primRef,3);

//       // z dir, bottom to top
//       addFace({hex[0],hex[1],hex[2],hex[3]},primRef,4);
//       addFace({hex[4],hex[7],hex[6],hex[5]},primRef,5);
//     }
    
//     void addFaces()
//     {
//       std::cout << "pushing " << prettyNumber(in->tets.size())
//                 << " tets (every dot is 100k)" << std::endl;
//       for (size_t i=0;i<in->tets.size();i++) {
//         if (!(i % 100000)) std::cout << "." << std::flush;
//         addFaces(in->tets[i],UMesh::PrimRef(UMesh::TET,i));
//       }
//       std::cout << std::endl;

//       std::cout << "pushing " << prettyNumber(in->pyrs.size()) << " pyrs" << std::endl;
//       for (size_t i=0;i<in->pyrs.size();i++)
//         addFaces(in->pyrs[i],UMesh::PrimRef(UMesh::PYR,i));
      
//       std::cout << "pushing " << prettyNumber(in->wedges.size()) << " wedges" << std::endl;
//       for (size_t i=0;i<in->wedges.size();i++)
//         addFaces(in->wedges[i],UMesh::PrimRef(UMesh::WEDGE,i));
      
//       std::cout << "pushing " << prettyNumber(in->hexes.size()) << " hexes" << std::endl;
//       for (size_t i=0;i<in->hexes.size();i++)
//         addFaces(in->hexes[i],UMesh::PrimRef(UMesh::HEX,i));
//     }
//     std::map<FaceKey,Face::SP> faces;
//     UMesh::SP in, out;
//     RemeshHelper indexer;
//   };

//   // UMesh::SP extractShell(UMesh::SP in)
//   // {
//   //   UMesh::SP shell = std::make_shared<UMesh>();

//   //   std::cout << "found shell w/ "
//   //             << prettyNumber(helper.faces.size()) << " faces (using ORIGINAL vertices from FULL tet-mesh)" << std::endl;

//   //   UMesh::SP mesh = std::make_shared<UMesh>();
    
//   //   return mesh;
//   // }
  

//   btm::Mesh::SP makeBTM(const ShellHelper &helper)
//   {
//     std::vector<vec3f> vertex;
//     std::vector<vec3f> normal;
//     std::vector<vec3f> color;
//     std::vector<vec2f> texcoord;
//     std::vector<vec3i> index;
//     std::vector<vec3f> triColor;

//     btm::Mesh::SP mesh = std::make_shared<btm::Mesh>();
//     mesh->vertex = helper.out->vertices;
//     for (auto f : helper.faces)
//       if (f.second->primsAndLocalFace.size() == 1) {
//         auto &indices = f.second->remeshedIndices;
//         switch (indices.size()) {
//         case 3: {
//           vec3i index(indices[0],indices[1],indices[2]);
//           mesh->index.push_back(index);
//         } break;
//         case 4: {
//           vec4i index(indices[0],indices[1],indices[2],indices[3]);
//           int lowest = arg_min(index);
//           mesh->index.push_back(vec3i(index[(lowest+0)%4],
//                                       index[(lowest+1)%4],
//                                       index[(lowest+2)%4]));
//           mesh->index.push_back(vec3i(index[(lowest+0)%4],
//                                       index[(lowest+2)%4],
//                                       index[(lowest+3)%4]));
//         } break;
//         default:
//           std::cout << "WARNING: found a face that is neither a triangle "
//                     << "nor a (bilinear) quad!? (#vertices = "
//                     << indices.size()  << std::endl;
//         };
//       }
//     return mesh;
//   }

//   void saveToBTM(const std::string &fileName,
//                  const UMesh::SP shell)
//   {
//     btm::Mesh::SP mesh = makeBTM(shell);
//     mesh->save(fileName);
//   }
  
//   void saveToOBJ(const std::string &objFileName,
//                  ShellHelper &helper)
//   {
//     std::ofstream obj(objFileName);
//     std::cout << "writing OBJ format to " << objFileName << std::endl;
//     std::cout << ".... THIS MAY TAKE A WHILE!" << std::endl;
//     for (auto v : helper.out->vertices)
//       obj << "v " << v.x << " " << v.y << " " << v.z << std::endl;

//     for (auto f : helper.faces) {
//       if (f.second->primsAndLocalFace.size() == 1) {
//         auto &indices = f.second->remeshedIndices;
//         obj << "f ";
//         for (auto i : indices)
//           obj << " " << (i+1);
//         obj << std::endl;
//       }
//       else if (f.second->primsAndLocalFace.size() != 2) {
//         auto &indices = f.second->remeshedIndices;
//         std::cout << "WARNING: face with num adjacent prims that's neither 1 nor 2!?" << std::endl;
//         std::cout << "face is ";
//         std::cout << "f ";
//         for (auto i : indices)
//           std::cout << " " << helper.out->vertices[i];
//         std::cout << std::endl;
//         std::cout << "shared by ";
//         for (auto i : f.second->primsAndLocalFace)
//           std::cout << " " << i.first.ID;
//         std::cout << std::endl;
//       }
//     }
//   }
    
//   extern "C" int main(int ac, char **av)
//   {
//     try {
//       std::string inFileName;
//       std::string outFileName;
//       std::string objFileName;
//       std::string btmFileName;

//       for (int i = 1; i < ac; i++) {
//         const std::string arg = av[i];
//         if (arg == "-o")
//           outFileName = av[++i];
//         else if (arg == "--obj" || arg == "-obj")
//           objFileName = av[++i];
//         else if (arg == "--tribin" || arg == "-tribin")
//           btmFileName = av[++i];
//         else if (arg[0] != '-')
//           inFileName = arg;
//         else {
//           throw std::runtime_error("./umeshComputeShell <in.umesh> -o <out.shell> [-tribin file.tribin] [--obj <out.obj]");
//         }
//       }
//       if (outFileName == "")
//         throw std::runtime_error("no output filename specified (-o)");

//       std::cout << "loading umesh from " << inFileName << std::endl;
//       UMesh::SP in = io::loadBinaryUMesh(inFileName);
//       if (!in->pyrs.empty() ||
//           !in->wedges.empty() ||
//           !in->hexes.empty())
//         std::cout << "Warning: umesh contains non-tet elements..." << std::endl;

//       ShellHelper helper(in);
//       std::cout << "saving tris and quads in umesh format to " << outFileName << std::endl;
//       io::saveBinaryUMesh(outFileName, helper.out);
//       std::cout << "done saving umesh file" << std::endl;
//       if (objFileName != "") {
//         saveToOBJ(objFileName, helper);
//       }
//       if (btmFileName != "") {
//         saveToBTM(btmFileName, helper.out);
//       }
//     }
//     catch (std::exception &e) {
//       std::cerr << "fatal error " << e.what() << std::endl;
//       exit(1);
//     }
//     return 0;
//   }  
// } // ::umesh
