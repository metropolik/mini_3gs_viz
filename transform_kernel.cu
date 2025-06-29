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

// Compute 2D covariance matrices and quad parameters for Gaussian splatting
extern "C" __global__
void compute_2d_covariance(const float* view_space_positions,    // View space positions (4 components each)
                          const float* scales,                  // Scale vectors (3 components each)
                          const float* rotations,               // Rotation quaternions (4 components each)
                          const float* mv_matrix,               // Model-view matrix (4x4)
                          const float* proj_matrix,             // Projection matrix (4x4)
                          float* cov2d_data,                    // Output: 2D covariance matrices (3 components: cov[0,0], cov[0,1], cov[1,1])
                          float* quad_params,                   // Output: Quad parameters (4 components: center_x, center_y, radius_x, radius_y)
                          int* visibility_mask,                 // Output: Visibility mask (1 if visible, 0 if culled)
                          int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Load Gaussian parameters
    int pos_offset = idx * 4;
    int scale_offset = idx * 3;
    int rot_offset = idx * 4;
    int cov_offset = idx * 3;
    int quad_offset = idx * 4;
    
    // Load view space position
    float vx = view_space_positions[pos_offset + 0];
    float vy = view_space_positions[pos_offset + 1];
    float vz = view_space_positions[pos_offset + 2];
    float vw = view_space_positions[pos_offset + 3];
    
    // Cull if behind camera (z > 0 in view space)
    if (vz > 0.0f) {
        visibility_mask[idx] = 0;
        return;
    }
    
    // Load scale and rotation
    float sx = scales[scale_offset + 0];
    float sy = scales[scale_offset + 1];
    float sz = scales[scale_offset + 2];
    
    float qw = rotations[rot_offset + 0];
    float qx = rotations[rot_offset + 1];
    float qy = rotations[rot_offset + 2];
    float qz = rotations[rot_offset + 3];
    
    // Build 3D covariance matrix: Σ_3D = R * S * S^T * R^T
    // First build rotation matrix R from quaternion
    float r11 = 1.0f - 2.0f * (qy*qy + qz*qz);
    float r12 = 2.0f * (qx*qy - qw*qz);
    float r13 = 2.0f * (qx*qz + qw*qy);
    float r21 = 2.0f * (qx*qy + qw*qz);
    float r22 = 1.0f - 2.0f * (qx*qx + qz*qz);
    float r23 = 2.0f * (qy*qz - qw*qx);
    float r31 = 2.0f * (qx*qz - qw*qy);
    float r32 = 2.0f * (qy*qz + qw*qx);
    float r33 = 1.0f - 2.0f * (qx*qx + qy*qy);
    
    // Compute R * S (rotation times scaling)
    float rs11 = r11 * sx, rs12 = r12 * sy, rs13 = r13 * sz;
    float rs21 = r21 * sx, rs22 = r22 * sy, rs23 = r23 * sz;
    float rs31 = r31 * sx, rs32 = r32 * sy, rs33 = r33 * sz;
    
    // Compute 3D covariance: (R*S) * (R*S)^T
    float cov3d_00 = rs11*rs11 + rs12*rs12 + rs13*rs13;
    float cov3d_01 = rs11*rs21 + rs12*rs22 + rs13*rs23;
    float cov3d_02 = rs11*rs31 + rs12*rs32 + rs13*rs33;
    float cov3d_11 = rs21*rs21 + rs22*rs22 + rs23*rs23;
    float cov3d_12 = rs21*rs31 + rs22*rs32 + rs23*rs33;
    float cov3d_22 = rs31*rs31 + rs32*rs32 + rs33*rs33;
    
    // Project to screen space using Jacobian of perspective projection
    // For point (x,y,z) in view space, screen coords are (x/z, y/z)
    // Jacobian J = [[1/z, 0, -x/z^2], [0, 1/z, -y/z^2]]
    float inv_z = 1.0f / (-vz);  // Note: vz is negative in view space
    float inv_z2 = inv_z * inv_z;
    
    // Jacobian matrix elements
    float j00 = inv_z;
    float j02 = vx * inv_z2;
    float j11 = inv_z;
    float j12 = vy * inv_z2;
    
    // Compute 2D covariance: Σ_2D = J * Σ_3D * J^T
    // Only need upper triangle since matrix is symmetric
    float cov2d_00 = j00*j00*cov3d_00 + j02*j02*cov3d_22 + 2.0f*j00*j02*cov3d_02;
    float cov2d_01 = j00*j11*cov3d_01 + j02*j12*cov3d_12 + j00*j12*cov3d_02 + j02*j11*cov3d_01;
    float cov2d_11 = j11*j11*cov3d_11 + j12*j12*cov3d_22 + 2.0f*j11*j12*cov3d_12;
    
    // Add a small regularization term to ensure positive definiteness
    cov2d_00 += 1e-6f;
    cov2d_11 += 1e-6f;
    
    // Store 2D covariance (symmetric, so store upper triangle)
    cov2d_data[cov_offset + 0] = cov2d_00;
    cov2d_data[cov_offset + 1] = cov2d_01;
    cov2d_data[cov_offset + 2] = cov2d_11;
    
    // Compute eigenvalues for quad sizing (3σ bounds)
    // For 2x2 symmetric matrix [[a,b],[b,c]], eigenvalues are:
    // λ = (a+c ± sqrt((a-c)² + 4b²)) / 2
    float trace = cov2d_00 + cov2d_11;
    float det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
    float discriminant = trace * trace - 4.0f * det;
    
    if (discriminant < 0.0f) {
        // Degenerate case, mark as invisible
        visibility_mask[idx] = 0;
        return;
    }
    
    float sqrt_disc = sqrtf(discriminant);
    float lambda1 = 0.5f * (trace + sqrt_disc);
    float lambda2 = 0.5f * (trace - sqrt_disc);
    
    // Compute radii (3σ = 3 * sqrt(eigenvalue))
    float radius_x = 3.0f * sqrtf(fmaxf(lambda1, 1e-6f));
    float radius_y = 3.0f * sqrtf(fmaxf(lambda2, 1e-6f));
    
    // Cull if quad would be too small (less than 1 pixel)
    if (radius_x < 1.0f || radius_y < 1.0f) {
        visibility_mask[idx] = 0;
        return;
    }
    
    // Project center to screen coordinates
    float center_x = vx * inv_z;
    float center_y = vy * inv_z;
    
    // Store quad parameters
    quad_params[quad_offset + 0] = center_x;
    quad_params[quad_offset + 1] = center_y;
    quad_params[quad_offset + 2] = radius_x;
    quad_params[quad_offset + 3] = radius_y;
    
    // Mark as visible
    visibility_mask[idx] = 1;
}