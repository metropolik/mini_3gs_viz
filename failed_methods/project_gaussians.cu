extern "C" __global__
void project_gaussians(
    const float* positions,      // (N, 3)
    const float* scales,         // (N, 3)
    const float* rotations,      // (N, 4) quaternions
    const float* view_matrix,    // (4, 4)
    const float* proj_matrix,    // (4, 4)
    float* depths,              // (N,) output depths for sorting
    float* cov2d,               // (N, 3) output 2D covariance (a, b, c) where cov = [[a, b], [b, c]]
    float* screen_pos,          // (N, 2) output screen positions
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Get position
    float x = positions[idx * 3 + 0];
    float y = positions[idx * 3 + 1];
    float z = positions[idx * 3 + 2];
    
    // Transform to view space
    float vx = view_matrix[0] * x + view_matrix[4] * y + view_matrix[8] * z + view_matrix[12];
    float vy = view_matrix[1] * x + view_matrix[5] * y + view_matrix[9] * z + view_matrix[13];
    float vz = view_matrix[2] * x + view_matrix[6] * y + view_matrix[10] * z + view_matrix[14];
    float vw = view_matrix[3] * x + view_matrix[7] * y + view_matrix[11] * z + view_matrix[15];
    
    depths[idx] = -vz / vw;  // Negative for proper sorting (closer = larger), divide by vw
    
    // Project to screen space
    float px = proj_matrix[0] * vx + proj_matrix[4] * vy + proj_matrix[8] * vz + proj_matrix[12] * vw;
    float py = proj_matrix[1] * vx + proj_matrix[5] * vy + proj_matrix[9] * vz + proj_matrix[13] * vw;
    float pz = proj_matrix[2] * vx + proj_matrix[6] * vy + proj_matrix[10] * vz + proj_matrix[14] * vw;
    float pw = proj_matrix[3] * vx + proj_matrix[7] * vy + proj_matrix[11] * vz + proj_matrix[15] * vw;
    
    // Debug: Store debug info for a non-zero point (idx 100)
    if (idx == 100) {
        // Store all transformation stages
        cov2d[0] = x;      // Original x
        cov2d[1] = y;      // Original y  
        cov2d[2] = z;      // Original z
        cov2d[3] = vx;     // View x (before division)
        cov2d[4] = vy;     // View y (before division)
        cov2d[5] = vz;     // View z (before division)
        cov2d[6] = vw;     // View w
        cov2d[7] = px;     // Projected x (before division)
        cov2d[8] = pw;     // Projected w
    }
    
    // Handle division by w carefully
    if (fabsf(pw) > 1e-6f) {
        screen_pos[idx * 2 + 0] = px / pw;
        screen_pos[idx * 2 + 1] = py / pw;
    } else {
        // This shouldn't happen with a proper projection matrix
        screen_pos[idx * 2 + 0] = 0.0f;
        screen_pos[idx * 2 + 1] = 0.0f;
        if (idx == 100) {
            printf("Warning: pw near zero for idx 100: %f\n", pw);
        }
    }
    
    // Simplified 2D covariance calculation (identity for now)
    // TODO: Proper 3D to 2D projection of covariance
    float sx = scales[idx * 3 + 0];
    float sy = scales[idx * 3 + 1];
    
    // Simple isotropic 2D covariance based on average scale
    float avg_scale = (sx + sy) * 0.5f;
    cov2d[idx * 3 + 0] = avg_scale * avg_scale;  // a
    cov2d[idx * 3 + 1] = 0.0f;                   // b
    cov2d[idx * 3 + 2] = avg_scale * avg_scale;  // c
}