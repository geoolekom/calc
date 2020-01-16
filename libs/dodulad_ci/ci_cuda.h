#include "ci.h"
#include <cuda_runtime_api.h>

namespace ci {
    __device__ void cudaIter(node_calc* ncData, size_t ncSize, float* f1, float* f2);
}
