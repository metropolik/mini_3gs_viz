extern "C" __global__
void transform_points(const float* mvp_matrix, const float* homogeneous_positions, 
                     float* transformed_positions, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Each point has 4 components (x, y, z, w)
    int point_offset = idx * 4;
    int output_offset = idx * 3;  // Output only x, y, z (3 components)
    
    // Load homogeneous position
    float x = homogeneous_positions[point_offset + 0];
    float y = homogeneous_positions[point_offset + 1];
    float z = homogeneous_positions[point_offset + 2];
    float w = homogeneous_positions[point_offset + 3];
    
    // Matrix multiplication: transformed = mvp * homogeneous_position
    // MVP matrix is 4x4, stored in row-major order
    float tx = mvp_matrix[0]  * x + mvp_matrix[1]  * y + mvp_matrix[2]  * z + mvp_matrix[3]  * w;
    float ty = mvp_matrix[4]  * x + mvp_matrix[5]  * y + mvp_matrix[6]  * z + mvp_matrix[7]  * w;
    float tz = mvp_matrix[8]  * x + mvp_matrix[9]  * y + mvp_matrix[10] * z + mvp_matrix[11] * w;
    float tw = mvp_matrix[12] * x + mvp_matrix[13] * y + mvp_matrix[14] * z + mvp_matrix[15] * w;
    
    // Perform perspective division
    float valid_w = (fabsf(tw) < 1e-8f) ? 1e-8f : tw;
    
    // Store transformed positions after perspective division (only x, y, z)
    transformed_positions[output_offset + 0] = tx / valid_w;
    transformed_positions[output_offset + 1] = ty / valid_w;
    transformed_positions[output_offset + 2] = tz / valid_w;
}