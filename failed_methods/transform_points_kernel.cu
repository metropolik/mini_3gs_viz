extern "C" __global__
void transform_points(
    const float* positions_in,   // (N, 3) input positions
    const float* view_matrix,    // (16) flattened 4x4 matrix (column-major)
    float* positions_out,        // (N, 3) output transformed positions
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Get input position
    float x = positions_in[idx * 3 + 0];
    float y = positions_in[idx * 3 + 1];
    float z = positions_in[idx * 3 + 2];
    
    // The view matrix seems to be stored with translation in the last row
    // This suggests we need to interpret it differently
    // Let's try treating the matrix as if each row represents the transformed basis
    
    // For a matrix stored row-major with translation in last row:
    // Row 0: X-axis transform
    // Row 1: Y-axis transform  
    // Row 2: Z-axis transform
    // Row 3: Translation + w
    
    float vx = view_matrix[0] * x + view_matrix[1] * y + view_matrix[2] * z + view_matrix[3] * 0.0f;
    float vy = view_matrix[4] * x + view_matrix[5] * y + view_matrix[6] * z + view_matrix[7] * 0.0f;
    float vz = view_matrix[8] * x + view_matrix[9] * y + view_matrix[10] * z + view_matrix[11] * 0.0f;
    
    // Add translation from last row
    vx += view_matrix[12];
    vy += view_matrix[13];
    vz += view_matrix[14];
    
    // Don't divide by w - view matrices typically have w=1
    positions_out[idx * 3 + 0] = vx;
    positions_out[idx * 3 + 1] = vy;
    positions_out[idx * 3 + 2] = vz;
}