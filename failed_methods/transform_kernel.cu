extern "C" __global__
void transform_points(
    const float* __restrict__ in_points,  // Input points (x, y, z)
    float* __restrict__ out_points,       // Output points (x, y, z)
    const float* __restrict__ transform,  // 4x4 transformation matrix (column-major, OpenGL format)
    const int num_points
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Load the point
    float x = in_points[idx * 3 + 0];
    float y = in_points[idx * 3 + 1];
    float z = in_points[idx * 3 + 2];
    float w = 1.0f;  // Homogeneous coordinate
    
    // Apply transformation (matrix is column-major, OpenGL format)
    // Matrix layout: [m00 m10 m20 m30] [m01 m11 m21 m31] [m02 m12 m22 m32] [m03 m13 m23 m33]
    //                    0   1   2   3     4   5   6   7     8   9  10  11    12  13  14  15
    float tx = transform[0] * x + transform[4] * y + transform[8]  * z + transform[12] * w;
    float ty = transform[1] * x + transform[5] * y + transform[9]  * z + transform[13] * w;
    float tz = transform[2] * x + transform[6] * y + transform[10] * z + transform[14] * w;
    float tw = transform[3] * x + transform[7] * y + transform[11] * z + transform[15] * w;
    
    // Perspective divide (if tw != 1)
    if (tw != 0.0f && tw != 1.0f) {
        tx /= tw;
        ty /= tw;
        tz /= tw;
    }
    
    // Store the result
    out_points[idx * 3 + 0] = tx;
    out_points[idx * 3 + 1] = ty;
    out_points[idx * 3 + 2] = tz;
}