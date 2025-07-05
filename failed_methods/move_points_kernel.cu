extern "C" __global__
void move_points(
    float* positions,    // (N, 3) positions to modify in-place
    float offset_x,      // Amount to move in x
    float offset_y,      // Amount to move in y
    float offset_z,      // Amount to move in z
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Move each point by the specified offsets
    positions[idx * 3 + 0] += offset_x;
    positions[idx * 3 + 1] += offset_y;
    positions[idx * 3 + 2] += offset_z;
}