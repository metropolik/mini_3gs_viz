// Include CUDA random number generation
#include <curand_kernel.h>

extern "C" __global__
void generate_random_points(
    float* positions,    // (N, 3) positions to fill with random values
    unsigned int seed,   // Random seed
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Initialize random state for this thread
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Generate random x,y,z coordinates in range [-1, 1]
    positions[idx * 3 + 0] = curand_uniform(&state) * 2.0f - 1.0f;  // x
    positions[idx * 3 + 1] = curand_uniform(&state) * 2.0f - 1.0f;  // y
    positions[idx * 3 + 2] = curand_uniform(&state) * 2.0f - 1.0f;  // z
}