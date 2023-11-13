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

/* computes the outer shell of a tet-mesh, ie, all the triangle and/or
   bilinear faces that are _not_ shared between two neighboring
   elements */ 

#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/io/btm/BTM.h"
#include "umesh/RemeshHelper.h"
#include <chrono>
#include <algorithm>

namespace umesh {

    struct ShellHelper
    {
        typedef std::vector<int> FaceKey;

        struct Face {
            typedef std::shared_ptr<Face> SP;
            std::vector<std::pair<UMesh::PrimRef, int>> primsAndLocalFace;

            /*! stores, for each vertex in this face, the list of indices AS
                THEY REFER TO THE ORIGINAL (VOLUMETRIC!) MESH's vertex
                array */
            std::vector<int> originalIndices;

            /*! stores, for each vertex in this face, the list of indices AS
              THEY REFER TO THE (SURFACE-ONLY, REMESHES) 'out' mesh */
            std::vector<int> remeshedIndices;
        };

        ShellHelper(UMesh::SP in)
            : in(in),
            out(std::make_shared<UMesh>()),
            indexer(*out)
        {
            addFaces();
        }

        inline
            FaceKey computeKey(std::vector<int> faceIndices)
        {
            std::sort(faceIndices.begin(), faceIndices.end());
            return faceIndices;
        }

        inline
            Face::SP findFace(const std::vector<int>& faceIndices)
        {
            FaceKey key = computeKey(faceIndices);
            auto it = faces.find(key);
            if (it != faces.end()) return it->second;

            Face::SP face = std::make_shared<Face>();
            face->originalIndices = faceIndices;

            for (auto org : faceIndices)
                face->remeshedIndices.push_back(indexer.getID(in->vertices[org],
                    (size_t)org));
            faces[key] = face;

            auto& idx = face->remeshedIndices;
            if (idx.size() == 3)
                out->triangles.push_back({ idx[0],idx[1],idx[2] });
            else
                out->quads.push_back({ idx[0],idx[1],idx[2],idx[3] });

            return face;
        }

        inline
            void addFace(const std::vector<int>& faceIndices,
                UMesh::PrimRef primRef,
                int localFaceID)
        {
            findFace(faceIndices)->primsAndLocalFace.push_back({ primRef,localFaceID });
        }

        inline
            void addFaces(const Tet& tet, UMesh::PrimRef primRef)
        {
            addFace({ tet[0],tet[1],tet[2] }, primRef, 3);
            addFace({ tet[0],tet[1],tet[3] }, primRef, 2);
            addFace({ tet[0],tet[2],tet[3] }, primRef, 1);
            addFace({ tet[1],tet[2],tet[3] }, primRef, 0);
        }

        void addFaces(const Pyr& pyr, UMesh::PrimRef primRef)
        {
            {
                addFace({ pyr[0],pyr[4],pyr[3] }, primRef, 0);
                addFace({ pyr[1],pyr[2],pyr[4] }, primRef, 1);
                addFace({ pyr[0],pyr[1],pyr[4] }, primRef, 2);
                addFace({ pyr[4],pyr[3],pyr[2] }, primRef, 3);

                addFace({ pyr[0],pyr[3],pyr[2],pyr[1] }, primRef, 4);
            }
        }

        void addFaces(const Wedge& wedge, UMesh::PrimRef primRef)
        {
            addFace({ wedge[0],wedge[3],wedge[5],wedge[2] }, primRef, 0);
            addFace({ wedge[1],wedge[4],wedge[5],wedge[2] }, primRef, 1);

            addFace({ wedge[0],wedge[1],wedge[2] }, primRef, 2);
            addFace({ wedge[3],wedge[5],wedge[4] }, primRef, 3);

            addFace({ wedge[0],wedge[1],wedge[4],wedge[3] }, primRef, 4);
        }
        void addFaces(const Hex& hex, UMesh::PrimRef primRef)
        {
            // windinroder.png: x dir, left to right
            addFace({ hex[0],hex[3],hex[7],hex[4] }, primRef, 0);
            addFace({ hex[1],hex[2],hex[6],hex[5] }, primRef, 1);

            // y dir, front to back
            addFace({ hex[0],hex[1],hex[5],hex[4] }, primRef, 2);
            addFace({ hex[2],hex[6],hex[7],hex[3] }, primRef, 3);

            // z dir, bottom to top
            addFace({ hex[0],hex[1],hex[2],hex[3] }, primRef, 4);
            addFace({ hex[4],hex[7],hex[6],hex[5] }, primRef, 5);
        }

        void addFaces()
        {
            std::cout << "pushing " << prettyNumber(in->tets.size())
                << " tets (every dot is 100k)" << std::endl;
            for (size_t i = 0; i < in->tets.size(); i++) {
                if (!(i % 100000)) std::cout << "." << std::flush;
                addFaces(in->tets[i], UMesh::PrimRef(UMesh::TET, i));
            }
            std::cout << std::endl;

            std::cout << "pushing " << prettyNumber(in->pyrs.size()) << " pyrs" << std::endl;
            for (size_t i = 0; i < in->pyrs.size(); i++)
                addFaces(in->pyrs[i], UMesh::PrimRef(UMesh::PYR, i));

            std::cout << "pushing " << prettyNumber(in->wedges.size()) << " wedges" << std::endl;
            for (size_t i = 0; i < in->wedges.size(); i++)
                addFaces(in->wedges[i], UMesh::PrimRef(UMesh::WEDGE, i));

            std::cout << "pushing " << prettyNumber(in->hexes.size()) << " hexes" << std::endl;
            for (size_t i = 0; i < in->hexes.size(); i++)
                addFaces(in->hexes[i], UMesh::PrimRef(UMesh::HEX, i));
        }
        std::map<FaceKey, Face::SP> faces;
        UMesh::SP in, out;
        RemeshHelper indexer;
    };

    extern "C" int main(int ac, char** av)
    {
        try {
            std::string inFileName;
            std::string outFileName;
            std::string objFileName;
            std::string btmFileName;

            for (int i = 1; i < ac; i++) {
                const std::string arg = av[i];
                if (arg == "-o")
                    outFileName = av[++i];
                else if (arg == "--obj" || arg == "-obj")
                    objFileName = av[++i];
                else if (arg == "--tribin" || arg == "-tribin")
                    btmFileName = av[++i];
                else if (arg[0] != '-')
                    inFileName = arg;
                else
                    throw std::runtime_error("./umeshComputeShell <in.umesh> -o <out.shell> [-tribin file.tribin] [--obj <out.obj]");
            }

            std::cout << "loading umesh from " << inFileName << std::endl;
            UMesh::SP in = io::loadBinaryUMesh(inFileName);
            if (!in->pyrs.empty() ||
                !in->wedges.empty() ||
                !in->hexes.empty())
                throw std::runtime_error("umesh contains non-tet elements...");

            std::cout << "computing all faces:" << std::endl;
            std::chrono::steady_clock::time_point
                begin = std::chrono::steady_clock::now();

            ShellHelper helper(in);
            std::chrono::steady_clock::time_point
                end = std::chrono::steady_clock::now();
            std::cout << "computed faces, found " << prettyNumber(helper.faces.size()) << " faces, took " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " secs" << std::endl;

            std::cout << "check: num vertices after re-indexing " << (helper.indexer.knownVertices.size()) << std::endl;;
        }
        catch (std::exception e) {
            std::cerr << "fatal error " << e.what() << std::endl;
        }
        return 0;
    }
} // ::umesh
