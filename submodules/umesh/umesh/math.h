// ======================================================================== //
// Copyright 2015-2020 Ingo Wald & Fabio Pellacini                          //
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

/*! this file taken and adapted from corresponding file in
  gitlab/ingowald/pbrtParser */

#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
# ifndef __PRETTY_FUNCTION__
#  define __PRETTY_FUNCTION__ __FUNCSIG__
# endif
#endif

#if defined(_MSC_VER)
#  define UMESH_DLL_EXPORT __declspec(dllexport)
#  define UMESH_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define UMESH_DLL_EXPORT __attribute__((visibility("default")))
#  define UMESH_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define UMESH_DLL_EXPORT
#  define UMESH_DLL_IMPORT
#endif

#if defined(UMESH_DLL_INTERFACE)
#  ifdef umesh_EXPORTS
#    define UMESH_INTERFACE UMESH_DLL_EXPORT
#  else
#    define UMESH_INTERFACE UMESH_DLL_IMPORT
#  endif
#else
#  define UMESH_INTERFACE /*static lib*/
#endif

#include <iostream>
#include <math.h> // using cmath causes issues under Windows
#include <cfloat>
#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits.h>

#if (!defined(__umesh_both__))
# if defined(__CUDA_ARCH__)
#  define __umesh_device   __device__
#  define __umesh_host     __host__
# else
#  define __umesh_device   /* ignore */
#  define __umesh_host     /* ignore */
# endif
# define __umesh_both__   __umesh_host __umesh_device
#endif
  
namespace umesh {
  struct range1f {
    range1f including(const float f) const;
    void extend(float f);
    void extend(const range1f &other);
    float lower = +FLT_MAX, upper = -FLT_MAX;
  };
  
  inline range1f range1f::including(const float f) const
  {
    range1f res;
    res.lower = std::min(lower,f);
    res.upper = std::max(upper,f);
    return res;
  }
  // inline range1f range1f::including(const range1f &other) const
  // { return { std::min(lower,other,lower),std::max(upper,other.upper) }; }
  
  inline void range1f::extend(float f)
  {
    lower = std::min(lower,f);
    upper = std::max(upper,f);
  }
  inline void range1f::extend(const range1f &other)
  {
    lower = std::min(lower,other.lower);
    upper = std::max(upper,other.upper);
  }

  struct vec3i;
  
  /*! namespace for syntax-only parser - this allows to distringuish
    high-level objects such as shapes from objects or transforms,
    but does *not* make any difference between what types of
    shapes, what their parameters mean, etc. Basically, at this
    level a triangle mesh is nothing but a shape that has a string
    with a given name, and parameters of given names and types */
  struct vec2f {
    typedef float scalar_t;
    vec2f() = default;
    explicit vec2f(float v) : x{v}, y{v} { }
    vec2f(float x, float y) : x{x}, y{y} { }
    float x, y;
  };
  struct vec3f {
    typedef float scalar_t;
    vec3f() = default;
    explicit vec3f(float v) : x{v}, y{v}, z{v} { }
    explicit vec3f(const vec3i &);
    vec3f(float x, float y, float z) : x{x}, y{y}, z{z} { }
    inline float &operator[](int i) { return (&x)[i]; }
    inline const float &operator[](int i) const { return (&x)[i]; }
    float x, y, z;
  };
  struct vec4f {
    typedef float scalar_t;
    vec4f() = default;
    vec4f(float v) : x{v}, y{v}, z{v}, w{v} { }
    vec4f(vec3f v, float w) : x{v.x}, y{v.y}, z{v.z}, w{w} { }
    vec4f(float x, float y, float z, float w) : x{x}, y{y}, z{z}, w{w} { }
    inline float &operator[](int i) { return (&x)[i]; }
    inline const float &operator[](int i) const { return (&x)[i]; }
    float x, y, z, w;
  };
  struct vec2i {
    int x, y;
    vec2i() = default;
    explicit vec2i(int v) : x{v}, y{v} { }
    vec2i(int x, int y) : x{x}, y{y} { }
    typedef int scalar_t;
  };
  struct vec3i {
    int x, y, z;
    vec3i() = default;
    vec3i(int v) : x{v}, y{v}, z{v} { }
    vec3i(int x, int y, int z) : x{x}, y{y}, z{z} { }
    typedef int scalar_t;
    inline int &operator[](int i) { return (&x)[i]; }
    inline const int &operator[](int i) const { return (&x)[i]; }
  };
  struct vec4i {
    int x, y, z, w;
    vec4i() = default;
    explicit vec4i(int v) : x{v}, y{v}, z{v}, w{v} { }
    vec4i(int x, int y, int z, int w) : x{x}, y{y}, z{z}, w{w} { }
    typedef int scalar_t;
    inline int &operator[](int i) { return (&x)[i]; }
    inline const int &operator[](int i) const { return (&x)[i]; }
  };
  struct mat3f {
    vec3f vx, vy, vz;
    mat3f() = default;
    explicit mat3f(const vec3f& v) : vx{v.x,0,0}, vy{0,v.y,0}, vz{0,0,v.z} { }
    mat3f(const vec3f& x, const vec3f& y, const vec3f& z) : vx{x}, vy{y}, vz{z} { }
  };
  struct affine3f {
    affine3f() = default;
    affine3f(const mat3f& l, const vec3f& p) :  l{l}, p{p} { }
    static affine3f identity() { return affine3f(mat3f(vec3f(1)), vec3f(0)); }
    static affine3f scale(const vec3f& u) { return affine3f(mat3f(u),vec3f(0)); }
    static affine3f translate(const vec3f& u) { return affine3f(mat3f(vec3f(1)), u); }
    static affine3f rotate(const vec3f& _u, float r);
    mat3f l;
    vec3f p;
  };
  struct box3f {
    vec3f lower, upper;
    box3f() : lower(FLT_MAX), upper(-FLT_MAX) {}//= default;
    box3f(const vec3f& lower, const vec3f& upper) : lower{lower}, upper{upper} { }
    inline bool  empty()  const
    { return (upper.x < lower.x) || (upper.y < lower.y) || (upper.z < lower.z); }
    static box3f empty_box() { return box3f(vec3f(FLT_MAX), vec3f(-FLT_MAX)); }
    box3f including(const vec3f &other) const;
    bool overlaps(const box3f &other) const;
    void extend(const vec3f& a);
    void extend(const box3f& a);
    vec3f size() const;
    vec3f center() const;
  };
  struct box3i {
    vec3i lower, upper;
    box3i() : lower(INT_MAX), upper(INT_MIN) {}//= default;
    box3i(const vec3i& lower, const vec3i& upper) : lower{lower}, upper{upper} { }
    inline bool  empty()  const
    { return (upper.x < lower.x) || (upper.y < lower.y) || (upper.z < lower.z); }
    static box3i empty_box() { return box3i(vec3i(INT_MAX), vec3i(INT_MIN)); }
    box3i including(const vec3i &other) const;
    bool overlaps(const box3i &other) const;
    void extend(const vec3i& a);
    void extend(const box3i& a);
    vec3i size() const;
    vec3i center() const;
  };
  struct box4f {
    vec4f lower, upper;
    box4f() : lower(FLT_MAX), upper(-FLT_MAX) {}//= default;
    box4f(const vec4f& lower, const vec4f& upper) : lower{lower}, upper{upper} { }
    inline bool  empty()  const
    { return (upper.x < lower.x) || (upper.y < lower.y) || (upper.z < lower.z); }
    static box3f empty_box() { return box3f(vec3f(FLT_MAX), vec3f(-FLT_MAX)); }
    void extend(const vec4f& a);
    void extend(const box4f& a);
  };

  
  typedef std::vector<std::pair<float, float>> pairNf;

  inline vec3f operator-(const vec3f& a) { return vec3f(-a.x, -a.y, -a.z); }
  inline vec3f operator-(const vec3f& a, const vec3f& b) { return vec3f(a.x - b.x, a.y - b.y, a.z - b.z); }
  inline vec3f operator+(const vec3f& a, const vec3f& b) { return vec3f(a.x + b.x, a.y + b.y, a.z + b.z); }
  inline vec4f operator+(const vec4f& a, const vec4f& b) { return vec4f(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

  inline vec3f operator*(const vec3f& a, float b) { return vec3f(a.x * b, a.y * b, a.z * b); }
  inline vec3f operator*(float a, const vec3f& b) { return vec3f(a * b.x, a * b.y, a * b.z); }
  inline vec3f operator*(const vec3f& a, const vec3f& b) { return vec3f(a.x * b.x, a.y * b.y, a.z * b.z); }
  inline mat3f operator*(const mat3f& a, float b) { return mat3f(a.vx * b, a.vy * b, a.vz * b); }
  inline vec3f operator*(const mat3f& a, const vec3f& b) { return a.vx * b.x + a.vy * b.y + a.vz * b.z; }
  inline mat3f operator*(const mat3f& a, const mat3f& b) { return mat3f(a * b.vx, a * b.vy, a * b.vz); }
  inline vec3f operator*(const affine3f& a, const vec3f& b) { return a.l * b + a.p; }


  inline vec3i operator-(const vec3i& a) { return vec3i(-a.x, -a.y, -a.z); }
  inline vec3i operator-(const vec3i& a, const vec3i& b) { return vec3i(a.x - b.x, a.y - b.y, a.z - b.z); }
  inline vec3i operator+(const vec3i& a, const vec3i& b) { return vec3i(a.x + b.x, a.y + b.y, a.z + b.z); }
  inline vec3i operator*(const vec3i& a, float b) { return vec3i(a.x * b, a.y * b, a.z * b); }
  inline vec3i operator*(float a, const vec3i& b) { return vec3i(a * b.x, a * b.y, a * b.z); }
  inline vec3i operator*(const vec3i& a, const vec3i& b) { return vec3i(a.x * b.x, a.y * b.y, a.z * b.z); }
  inline vec3i operator>>(const vec3i& a, int b) { return vec3i(a.x>>b, a.y>>b, a.z>>b); }
  inline vec3i operator/(const vec3i& a, int b) { return vec3i(a.x/b, a.y/b, a.z/b); }


  inline affine3f operator*(const affine3f& a, const affine3f& b) { return affine3f(a.l * b.l, a.l * b.p + a.p); }

  inline float dot(const vec3f& a, const vec3f& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
  inline vec3f cross(const vec3f& a, const vec3f& b) { return vec3f(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
  inline vec3f normalize(const vec3f& a) { return a * (1 / sqrtf(dot(a,a))); }

  inline mat3f transpose(const mat3f& a) { return mat3f(vec3f(a.vx.x, a.vy.x, a.vz.x), vec3f(a.vx.y, a.vy.y, a.vz.y), vec3f(a.vx.z, a.vy.z, a.vz.z)); }
  inline float determinant(const mat3f& a) { return dot(a.vx,cross(a.vy,a.vz)); }
  inline mat3f adjoint_transpose(const mat3f& a) { return mat3f(cross(a.vy,a.vz),cross(a.vz,a.vx),cross(a.vx,a.vy)); }
  inline mat3f inverse_transpose(const mat3f& a) { return adjoint_transpose(a) * (1 / determinant(a)); }
  inline mat3f inverse(const mat3f& a) {return transpose(inverse_transpose(a)); }
  inline affine3f inverse(const affine3f& a) { mat3f il = inverse(a.l); return affine3f(il,-(il*a.p)); }

  inline vec3f xfmPoint(const affine3f& m, const vec3f& p) { return m * p; }
  inline vec3f xfmVector(const affine3f& m, const vec3f& v) { return m.l * v; }
  inline vec3f xfmNormal(const affine3f& m, const vec3f& n) { return inverse_transpose(m.l) * n; }

  inline vec4f operator*(const vec4f& a, const vec4f& b) { return vec4f(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }


  inline affine3f affine3f::rotate(const vec3f& _u, float r) {
    vec3f u = normalize(_u);
    float s = sinf(r), c = cosf(r);
    mat3f l = mat3f(vec3f(u.x*u.x+(1-u.x*u.x)*c, u.x*u.y*(1-c)+u.z*s, u.x*u.z*(1-c)-u.y*s),
                    vec3f(u.x*u.y*(1-c)-u.z*s, u.y*u.y+(1-u.y*u.y)*c, u.y*u.z*(1-c)+u.x*s),   
                    vec3f(u.x*u.z*(1-c)+u.y*s, u.y*u.z*(1-c)-u.x*s, u.z*u.z+(1-u.z*u.z)*c));
    return affine3f(l, vec3f(0));
  }

  inline int arg_max(const vec3f &v) {
    return (v.x>v.y)
      ? ((v.x>v.z)?0:2)
      : ((v.y>v.z)?1:2);
  }
  inline int arg_min(const vec3f &v) {
    return (v.x<v.y)
      ? ((v.x<v.z)?0:2)
      : ((v.y<v.z)?1:2);
  }
  inline int arg_min(const vec3i &v) {
    return (v.x<v.y)
      ? ((v.x<v.z)?0:2)
      : ((v.y<v.z)?1:2);
  }
  inline int arg_min(const vec4i &v) {
    int sd   = 0;
    int sv = v.x;
    if (v.y < sv) { sv = v.y; sd = 1; }
    if (v.z < sv) { sv = v.z; sd = 2; }
    if (v.w < sv) { sv = v.w; sd = 3; }
    return sd;
  }
  inline float reduce_min(const vec3f v)
  { return std::min(std::min(v.x,v.y),v.z); }
  inline float max(float a, float b) { return a<b ? b:a; }
  inline float min(float a, float b) { return a>b ? b:a; }
  inline int max(int a, int b) { return a<b ? b:a; }
  inline int min(int a, int b) { return a>b ? b:a; }
  inline vec3f max(const vec3f& a, const vec3f& b) { return vec3f(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z)); }
  inline vec3f min(const vec3f& a, const vec3f& b) { return vec3f(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z)); }
  inline vec3i max(const vec3i& a, const vec3i& b) { return vec3i(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z)); }
  inline vec3i min(const vec3i& a, const vec3i& b) { return vec3i(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z)); }
  inline void box3f::extend(const vec3f& p) { lower=min(lower,p); upper=max(upper,p); }
  inline void box3i::extend(const vec3i& p) { lower=min(lower,p); upper=max(upper,p); }
  inline void box3f::extend(const box3f& b) { lower=min(lower,b.lower); upper=max(upper,b.upper); }
  inline void box3i::extend(const box3i& b) { lower=min(lower,b.lower); upper=max(upper,b.upper); }

  inline bool box3f::overlaps(const box3f &other) const
  {
    return
      !(lower.x > other.upper.x) &&
      !(lower.y > other.upper.y) &&
      !(lower.z > other.upper.z) &&
      !(upper.x < other.lower.x) &&
      !(upper.y < other.lower.y) &&
      !(upper.z < other.lower.z);
  }

  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }
  inline size_t divRoundUp(size_t a, size_t b) { return (a + b - 1) / b; }
  inline box3f intersection(const box3f &a, const box3f &b)
  { return {max(a.lower,b.lower),min(a.upper,b.upper)}; }
  inline box3f box3f::including(const vec3f &other) const
  { return box3f{min(lower,other),max(upper,other)}; }
  inline vec3f box3f::size() const { return upper - lower; }
  inline vec3i box3i::size() const { return upper - lower; }
  inline vec3f box3f::center() const { return 0.5f*(lower+upper); }

  inline std::ostream& operator<<(std::ostream& o, const range1f& v) { return o << "[" << v.lower << ".." << v.upper << "]"; }
  inline std::ostream& operator<<(std::ostream& o, const vec2f& v) { return o << "(" << v.x << "," << v.y << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const vec3f& v) { return o << "(" << v.x << "," << v.y << "," << v.z << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const vec3i& v) { return o << "(" << v.x << "," << v.y << "," << v.z << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const vec4f& v) { return o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const vec2i& v) { return o << "(" << v.x << "," << v.y << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const vec4i& v) { return o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")"; }
  inline std::ostream& operator<<(std::ostream& o, const mat3f& m) { return o << "{ vx = " << m.vx << ", vy = " << m.vy << ", vz = " << m.vz << "}"; }
  inline std::ostream& operator<<(std::ostream& o, const affine3f& m) { return o << "{ l = " << m.l << ", p = " << m.p << " }"; }
  inline std::ostream& operator<<(std::ostream &o, const box3f& b) { return o << "[" << b.lower <<":"<<b.upper<<"]"; }
  inline std::ostream& operator<<(std::ostream &o, const box3i& b) { return o << "[" << b.lower <<":"<<b.upper<<"]"; }

  inline bool operator==(const vec2f &a, const vec2f &b)
  { return a.x==b.x && a.y==b.y; }
  inline bool operator!=(const vec2f &a, const vec2f &b)
  { return !(a.x==b.x && a.y==b.y); }
  inline bool operator==(const vec3f &a, const vec3f &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z; }
  inline bool operator!=(const vec3f &a, const vec3f &b)
  { return !(a.x==b.x && a.y==b.y && a.z==b.z); }
  inline bool operator==(const vec3i &a, const vec3i &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z; }
  inline bool operator!=(const vec3i &a, const vec3i &b)
  { return !(a.x==b.x && a.y==b.y && a.z==b.z); }
  
  inline bool operator==(const vec4f &a, const vec4f &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
  inline bool operator!=(const vec4f &a, const vec4f &b)
  { return !(a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w); }
  
  inline bool operator<(const vec3f &a, const vec3f &b)
  {
    return (a.x<b.x) || ((a.x==b.x) && ((a.y<b.y) || ((a.y==b.y) && (a.z<b.z))));
  }
  inline bool operator<(const vec3i &a, const vec3i &b)
  {
#if 1
    const uint64_t la = (const uint64_t &)a;
    const uint64_t lb = (const uint64_t &)b;
    return ((la < lb) || (la==lb && (const uint32_t&)a.z < (const uint32_t&)b.z));
#else
    return
      (a.x<b.x) ||
      (a.x==b.x && a.y<b.y) ||
      (a.x==b.x && a.y==b.y && a.z<b.z);
#endif
  }
  inline bool operator<(const vec4i &a, const vec4i &b)
  {
    return
      (a.x<b.x) ||
      (a.x==b.x && a.y<b.y) ||
      (a.x==b.x && a.y==b.y && a.z<b.z) ||
      (a.x==b.x && a.y==b.y && a.z==b.z && a.w<b.w);
  }
  inline bool operator<(const vec4f &a, const vec4f &b)
  {
    return
      (a.x<b.x) ||
      (a.x==b.x && a.y<b.y) ||
      (a.x==b.x && a.y==b.y && a.z<b.z) ||
      (a.x==b.x && a.y==b.y && a.z==b.z && a.w<b.w);
  }


#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  
  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyDouble(const double val) {
    const double absVal = abs(val);
    char result[1000];

    if      (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e18f,'E');
    else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e15f,'P');
    else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",val/1e12f,'T');
    else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",val/1e09f,'G');
    else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",val/1e06f,'M');
    else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",val/1e03f,'k');
    else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",val*1e15f,'f');
    else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",val*1e12f,'p');
    else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",val*1e09f,'n');
    else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",val*1e06f,'u');
    else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",val*1e03f,'m');
    else osp_snprintf(result,1000,"%f",(float)val);
    return result;
  }

  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyNumber(const size_t s) {
    const double val = (double)s;
    char result[1000];

    if      (val >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e18f,'E');
    else if (val >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e15f,'P');
    else if (val >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",val/1e12f,'T');
    else if (val >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",val/1e09f,'G');
    else if (val >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",val/1e06f,'M');
    else if (val >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",val/1e03f,'k');
    else osp_snprintf(result,1000,"%zu",s);
    return result;
  }
#undef osp_snprintf
  
  inline vec3f::vec3f(const vec3i &v) : x{(float)v.x}, y{(float)v.y}, z{(float)v.z} { }

  inline float length(const vec3f &v) { return sqrtf(dot(v,v)); }

} // ::pbrt
