extern "C" __global__
void transform_points(const float* matrix, const float* input_positions, 
                     float* output_positions, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Input has 4 components (x, y, z, w)
    int input_offset = idx * 4;
    
    // Load input position
    float x = input_positions[input_offset + 0];
    float y = input_positions[input_offset + 1];
    float z = input_positions[input_offset + 2];
    float w = input_positions[input_offset + 3];
    
    // Matrix multiplication: transformed = matrix * input_position
    // Matrix is 4x4, stored in row-major order
    float tx = matrix[0]  * x + matrix[1]  * y + matrix[2]  * z + matrix[3]  * w;
    float ty = matrix[4]  * x + matrix[5]  * y + matrix[6]  * z + matrix[7]  * w;
    float tz = matrix[8]  * x + matrix[9]  * y + matrix[10] * z + matrix[11] * w;
    float tw = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w;
    
    // Determine output format based on output buffer size
    // If output has 4 components per point, keep homogeneous (no perspective division)
    // If output has 3 components per point, perform perspective division
    
    // Check if we should perform perspective division by examining if output stride is 3 or 4
    // We can infer this by checking the memory layout pattern
    // For now, we'll use a simple heuristic: if the kernel is called with 4-component output,
    // we assume it's the first pass (MV transformation), otherwise it's the second pass (P transformation)
    
    // We need to determine output format - let's assume homogeneous output for now
    // and create a separate kernel for perspective division
    int output_offset = idx * 4;  // Assume 4-component output for now
    
    output_positions[output_offset + 0] = tx;
    output_positions[output_offset + 1] = ty;
    output_positions[output_offset + 2] = tz;
    output_positions[output_offset + 3] = tw;
}

extern "C" __global__
void transform_points_with_perspective(const float* matrix, const float* input_positions, 
                                      float* output_positions, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Input has 4 components (x, y, z, w)
    int input_offset = idx * 4;
    int output_offset = idx * 3;  // Output only x, y, z (3 components)
    
    // Load input position
    float x = input_positions[input_offset + 0];
    float y = input_positions[input_offset + 1];
    float z = input_positions[input_offset + 2];
    float w = input_positions[input_offset + 3];
    
    // Matrix multiplication: transformed = matrix * input_position
    // Matrix is 4x4, stored in row-major order
    float tx = matrix[0]  * x + matrix[1]  * y + matrix[2]  * z + matrix[3]  * w;
    float ty = matrix[4]  * x + matrix[5]  * y + matrix[6]  * z + matrix[7]  * w;
    float tz = matrix[8]  * x + matrix[9]  * y + matrix[10] * z + matrix[11] * w;
    float tw = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w;
    
    // Perform perspective division
    float valid_w = (fabsf(tw) < 1e-8f) ? 1e-8f : tw;
    
    // Store transformed positions after perspective division (only x, y, z)
    output_positions[output_offset + 0] = tx / valid_w;
    output_positions[output_offset + 1] = ty / valid_w;
    output_positions[output_offset + 2] = tz / valid_w;
}