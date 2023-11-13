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

// ours:
#include "UMesh.h"
#include "io/UMesh.h"
#include "io/IO.h"
#include <sstream>


#ifndef PRINT
#ifdef __CUDA_ARCH__
# define PRINT(va) /**/
# define PING /**/
#else
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif
#endif

namespace umesh {
  
  const size_t bum_magic = 0x234235567ULL;
  
  /*! can be used to turn on/off logging/diagnostic messages in entire
    umesh library */
  bool verbose = false;

  /*! helper functoin for printf debugging - puts the four elemnt
    sizes into a human-readable (short) string*/
  std::string sizeString(UMesh::SP mesh)
  {
    std::stringstream str;
    str << "v:" << prettyNumber(mesh->vertices.size())
        << ",t:" << prettyNumber(mesh->tets.size())
        << ",p:" << prettyNumber(mesh->pyrs.size())
        << ",w:" << prettyNumber(mesh->wedges.size())
        << ",h:" << prettyNumber(mesh->hexes.size());
    return str.str();
  }
  

  /*! write - binary - to given (bianry) stream */
  void UMesh::writeTo(std::ostream &out) const
  {
    io::writeElement(out,bum_magic);
    io::writeVector(out,vertices);

    // iw - changed that to write 'numPerVertex' rather than always
    // write one - this allows for later switching to more than one
    // attribute
    if (perVertex) {
      size_t numPerVertexAttributes = 1;
      io::writeElement(out,numPerVertexAttributes);
      io::writeString(out,perVertex->name);
      io::writeVector(out,perVertex->values);
    } else {
      size_t numPerVertexAttributes = 0;
      io::writeElement(out,numPerVertexAttributes);
    }
    size_t numPerElementAttributes = 0;
    io::writeElement(out,numPerElementAttributes);
    
    io::writeVector(out,triangles);
    io::writeVector(out,quads);
    io::writeVector(out,tets);
    io::writeVector(out,pyrs);
    io::writeVector(out,wedges);
    io::writeVector(out,hexes);
    io::writeVector(out,vertexTag);
  }
  
  /*! write - binary - to given file */
  void UMesh::saveTo(const std::string &fileName) const
  {
    std::ofstream out(fileName, std::ios_base::binary);
    writeTo(out);
  }
    
  /*! read from given (binary) stream */
  void UMesh::readFrom(std::istream &in)
  {
    const size_t bum_magic_old = 0x234235566ULL;
    bool supportsMultipleAttributes = true;
    size_t magic;
    io::readElement(in,magic);
    if (magic != bum_magic)
    {
      if (magic != bum_magic_old)
        throw std::runtime_error("wrong magic number in umesh file ...");
      supportsMultipleAttributes = false;
    }
    io::readVector(in,this->vertices,"vertices");
    size_t numPerVertexAttributes = 1;
    if (supportsMultipleAttributes)
      io::readElement(in,numPerVertexAttributes);
    if (numPerVertexAttributes) {
      this->perVertex = std::make_shared<Attribute>();
      if (supportsMultipleAttributes)
        io::readString(in,perVertex->name);
      io::readVector(in,perVertex->values,"scalars");
      this->perVertex->finalize();
    }
      
    size_t numPerElementAttributes = 0;
    if (supportsMultipleAttributes)
      io::readElement(in,numPerElementAttributes);
    assert(numPerElementAttributes == 0);
    
    io::readVector(in,this->triangles,"triangles");
    io::readVector(in,this->quads,"quads");
    io::readVector(in,this->tets,"tets");
    io::readVector(in,this->pyrs,"pyramids");
    io::readVector(in,this->wedges,"wedges");
    io::readVector(in,this->hexes,"hexes");
    // try {
    if (!in.eof())
      try {
        io::readVector(in,this->vertexTag,"vertexTags");
      } catch (...) {
        /* ignore ... */
      }
  
    this->finalize();
  }

  /*! read from given file, assuming file format as used by saveTo() */
  UMesh::SP UMesh::loadFrom(const std::string &fileName)
  {
    UMesh::SP mesh = std::make_shared<UMesh>();
    std::ifstream in(fileName, std::ios_base::binary);
    if (!in.good())
      throw std::runtime_error("#umesh: could not open '"+fileName+"'");
    mesh->readFrom(in);
    return mesh;
  }
  

  /*! tells this attribute that its values are set, and precomputations can be done */
  void Attribute::finalize()
  {
    std::mutex mutex;
    parallel_for_blocked
      (0,values.size(),16*1024,
       [&](size_t begin, size_t end) {
         range1f rangeValueRange;
         for (size_t i=begin;i<end;i++)
           rangeValueRange.extend(values[i]);
         std::lock_guard<std::mutex> lock(mutex);
         valueRange.extend(rangeValueRange.lower);
         valueRange.extend(rangeValueRange.upper);
       });
  }


  /*! create std::vector of primitmive references (bounding box plus
    tag) for every volumetric prim in this mesh */
  std::vector<UMesh::PrimRef> UMesh::createVolumePrimRefs()
  {
    std::vector<PrimRef> result;
    createVolumePrimRefs(result);
    return result;
    // return std::move(result);
  }


  /*! create std::vector of all primrefs for all _surface_ elements
    (triangles and quads) */
  std::vector<UMesh::PrimRef> UMesh::createSurfacePrimRefs()
  {
    std::vector<PrimRef> result;
    createSurfacePrimRefs(result);
    return (result);
    // return std::move(result);
  }
    

  /*! create std::vector of all primrefs for all _surface_ elements
    (triangles and quads) */
  void UMesh::createSurfacePrimRefs(std::vector<PrimRef> &result)
  {
    result.resize(triangles.size()+quads.size());
    parallel_for_blocked
      (0ull,triangles.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[i] = PrimRef(TRI,i);
       });
    parallel_for_blocked
      (0ull,quads.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[triangles.size()+i] = PrimRef(QUAD,i);
       });
  }
  
  /*! create std::vector of primitmive references (bounding box plus
    tag) for every volumetric prim in this mesh */
  void UMesh::createVolumePrimRefs(std::vector<UMesh::PrimRef> &result)
  {
    result.resize(tets.size()+pyrs.size()+wedges.size()+hexes.size());
    parallel_for_blocked
      (0ull,tets.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[i] = PrimRef(TET,i);
       });
    parallel_for_blocked
      (0ull,pyrs.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[tets.size()+i] = PrimRef(PYR,i);
       });
    parallel_for_blocked
      (0ull,wedges.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[tets.size()+pyrs.size()+i] = PrimRef(WEDGE,i);
       });
    parallel_for_blocked
      (0ull,hexes.size(),64*1024,
       [&](size_t begin,size_t end){
         for (size_t i=begin;i<end;i++)
           result[tets.size()+pyrs.size()+wedges.size()+i] = PrimRef(HEX,i);
       });
  }


  /*! finalize a mesh, and compute min/max ranges where required */
  void UMesh::finalize()
  {
    if (perVertex) perVertex->finalize();
    bounds = box3f();
    std::mutex mutex;
    parallel_for_blocked
      (0,vertices.size(),16*1024,
       [&](size_t begin, size_t end) {
         box3f rangeBounds;
         for (size_t i=begin;i<end;i++)
           rangeBounds.extend(vertices[i]);
         std::lock_guard<std::mutex> lock(mutex);
         bounds.extend(rangeBounds);
       });
  }
  
  /*! print some basic info of this mesh to std::cout */
  void UMesh::print()
  {
    std::cout << toString(false);
  }
  
  /*! return a string of the form "UMesh{#tris=...}" */
  std::string UMesh::toString(bool compact) const
  {
    std::stringstream ss;

    if (compact) {
      ss << "UMesh(";
      ss << "#verts=" << prettyNumber(vertices.size());
      if (!triangles.empty())
        ss << ",#tris=" << prettyNumber(triangles.size());
      if (!quads.empty())
        ss << ",#quads=" << prettyNumber(quads.size());
      if (!tets.empty())
        ss << ",#tets=" << prettyNumber(tets.size());
      if (!pyrs.empty())
        ss << ",#pyrs=" << prettyNumber(pyrs.size());
      if (!wedges.empty())
        ss << ",#wedges=" << prettyNumber(wedges.size());
      if (!hexes.empty())
        ss << ",#hexes=" << prettyNumber(hexes.size());
      if (perVertex) {
        ss << ",scalars=yes(name='" << perVertex->name << "')";
      } else {
        ss << ",scalars=no";
      }
      ss << ")";
    } else {
      ss << "#verts : " << prettyNumber(vertices.size()) << std::endl;
      ss << "#tris  : " << prettyNumber(triangles.size()) << std::endl;
      ss << "#quads : " << prettyNumber(quads.size()) << std::endl;
      ss << "#tets  : " << prettyNumber(tets.size()) << std::endl;
      ss << "#pyrs  : " << prettyNumber(pyrs.size()) << std::endl;
      ss << "#wedges: " << prettyNumber(wedges.size()) << std::endl;
      ss << "#hexes : " << prettyNumber(hexes.size()) << std::endl;
      if  (!bounds.empty())
        ss << "bounds : " << bounds << std::endl;
      if (perVertex) {
        range1f valueRange = getValueRange();
        if (valueRange.lower > valueRange.upper)
          ss << "values : yes (range not yet computed)" << std::endl;
        else
          ss << "values : " << valueRange << std::endl;
      } else
        ss << "values : <none>" << std::endl;
    }
    return ss.str();
  }

} // ::tetty
