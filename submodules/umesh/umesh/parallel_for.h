// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#pragma once

// std
#include <mutex>

#ifdef UMESH_DISABLE_TBB
# undef UMESH_HAVE_TBB
#endif

// tbb
#if UMESH_HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#define UMESH_HAVE_PARALLEL_FOR 1
#endif

namespace umesh {

  template<typename INDEX_T, typename TASK_T>
  inline void serial_for(INDEX_T nTasks, TASK_T&& taskFunction)
  {
    for (INDEX_T taskIndex = 0; taskIndex < nTasks; ++taskIndex) {
      taskFunction(taskIndex);
    }
  }
  
#if UMESH_HAVE_TBB
  template<typename INDEX_T, typename TASK_T>
  inline void parallel_for(INDEX_T nTasks, TASK_T&& taskFunction, size_t blockSize=1)
  {
    if (nTasks == 0) return;
    if (nTasks == 1)
      taskFunction(size_t(0));
    else if (blockSize==1) {
      tbb::parallel_for(INDEX_T(0), nTasks, std::forward<TASK_T>(taskFunction));
    } else {
      const size_t numBlocks = (nTasks+blockSize-1)/blockSize;
      tbb::parallel_for((size_t)0, numBlocks, [&](size_t blockIdx){
                                                size_t begin = blockIdx*blockSize;
                                                size_t end   = std::min(begin+blockSize,size_t(nTasks));
                                                for (size_t i=begin;i<end;i++)
                                                  taskFunction(INDEX_T(i));
                                              });
    }
  }
#else
#ifndef UMESH_DISABLE_TBB
// # pragma message("(note): cmake did not find TBB, will replace all parallel_for's with serial_for's")
#endif
  template<typename INDEX_T, typename TASK_T>
  inline void parallel_for(INDEX_T nTasks, TASK_T&& taskFunction, size_t blockSize=1)
  { serial_for(nTasks,taskFunction); }
#endif

  template<typename TASK_T>
  void serial_for_blocked(size_t begin, size_t end, size_t blockSize,
                          TASK_T &&taskFunction)
  {
    for (size_t block_begin=begin; block_begin < end; block_begin += blockSize)
      taskFunction(block_begin,std::min(block_begin+blockSize,end));
  }
  
  template<typename TASK_T>
  void parallel_for_blocked(size_t begin, size_t end, size_t blockSize,
                            const TASK_T &taskFunction)
  {
    const size_t numTasks = end-begin;
    const size_t numBlocks = (numTasks+blockSize-1)/blockSize;
    parallel_for(numBlocks,[&](size_t blockID){
                             size_t block_begin = begin+blockID*blockSize;
                             taskFunction(block_begin,std::min(block_begin+blockSize,end));
                           });
  }
  
} // ::umesh
