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

// Device function to project a point from view space to NDC
__device__ void project_to_ndc(float vx, float vy, float vz, float vw,
                               const float* proj_matrix,
                               float& ndc_x, float& ndc_y, float& ndc_z) {
    // Apply projection matrix
    float clip_x = proj_matrix[0] * vx + proj_matrix[1] * vy + proj_matrix[2] * vz + proj_matrix[3] * vw;
    float clip_y = proj_matrix[4] * vx + proj_matrix[5] * vy + proj_matrix[6] * vz + proj_matrix[7] * vw;
    float clip_z = proj_matrix[8] * vx + proj_matrix[9] * vy + proj_matrix[10] * vz + proj_matrix[11] * vw;
    float clip_w = proj_matrix[12] * vx + proj_matrix[13] * vy + proj_matrix[14] * vz + proj_matrix[15] * vw;
    
    // Perform perspective division
    float inv_w = 1.0f / (fabsf(clip_w) < 1e-8f ? 1e-8f : clip_w);
    ndc_x = clip_x * inv_w;
    ndc_y = clip_y * inv_w;
    ndc_z = clip_z * inv_w;
}

// Compute 2D covariance matrices and quad parameters for Gaussian splatting
extern "C" __global__
void compute_2d_covariance(const float* view_space_positions,    // View space positions (4 components each)
                          const float* scales,                  // Scale vectors (3 components each)
                          const float* rotations,               // Rotation quaternions (4 components each)
                          const float* mv_matrix,               // Model-view matrix (4x4)
                          const float* proj_matrix,             // Projection matrix (4x4)
                          float* cov2d_data,                    // Output: 2D covariance matrices (3 components: cov[0,0], cov[0,1], cov[1,1])
                          float* quad_params,                   // Output: Quad parameters (5 components: center_x, center_y, radius_x, radius_y, ndc_z)
                          int* visibility_mask,                 // Output: Visibility mask (1 if visible, 0 if culled)
                          float viewport_width,                 // Viewport width in pixels
                          float viewport_height,                // Viewport height in pixels
                          int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Load Gaussian parameters
    int pos_offset = idx * 4;
    int scale_offset = idx * 3;
    int rot_offset = idx * 4;
    int cov_offset = idx * 3;
    int quad_offset = idx * 5;
    
    // Load view space position
    float vx = view_space_positions[pos_offset + 0];
    float vy = view_space_positions[pos_offset + 1];
    float vz = view_space_positions[pos_offset + 2];
    float vw = view_space_positions[pos_offset + 3];
    
    // Don't cull for performance - render all quads
    // Even if behind camera, still compute parameters
    visibility_mask[idx] = 1;  // Mark all as visible
    
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
    
    // Apply viewing transformation W to the 3D covariance: Σ' = W * Σ_3D * W^T
    // Extract the 3x3 rotation part of the model-view matrix (upper-left 3x3)
    float w00 = mv_matrix[0], w01 = mv_matrix[1], w02 = mv_matrix[2];
    float w10 = mv_matrix[4], w11 = mv_matrix[5], w12 = mv_matrix[6];
    float w20 = mv_matrix[8], w21 = mv_matrix[9], w22 = mv_matrix[10];
    
    // Compute W * Σ_3D
    float ws00 = w00*cov3d_00 + w01*cov3d_01 + w02*cov3d_02;
    float ws01 = w00*cov3d_01 + w01*cov3d_11 + w02*cov3d_12;
    float ws02 = w00*cov3d_02 + w01*cov3d_12 + w02*cov3d_22;
    float ws10 = w10*cov3d_00 + w11*cov3d_01 + w12*cov3d_02;
    float ws11 = w10*cov3d_01 + w11*cov3d_11 + w12*cov3d_12;
    float ws12 = w10*cov3d_02 + w11*cov3d_12 + w12*cov3d_22;
    float ws20 = w20*cov3d_00 + w21*cov3d_01 + w22*cov3d_02;
    float ws21 = w20*cov3d_01 + w21*cov3d_11 + w22*cov3d_12;
    float ws22 = w20*cov3d_02 + w21*cov3d_12 + w22*cov3d_22;
    
    // Compute (W * Σ_3D) * W^T to get view space covariance
    float cov_view_00 = ws00*w00 + ws01*w01 + ws02*w02;
    float cov_view_01 = ws00*w10 + ws01*w11 + ws02*w12;
    float cov_view_02 = ws00*w20 + ws01*w21 + ws02*w22;
    float cov_view_11 = ws10*w10 + ws11*w11 + ws12*w12;
    float cov_view_12 = ws10*w20 + ws11*w21 + ws12*w22;
    float cov_view_22 = ws20*w20 + ws21*w21 + ws22*w22;
    
    // Project to screen space using Jacobian of perspective projection
    // The projection matrix transforms (x,y,z) to (fx*x/z, fy*y/z) where fx, fy are from proj matrix
    // Extract focal length scaling from projection matrix
    float fx = proj_matrix[0];   // P[0,0] = focal_x / aspect
    float fy = proj_matrix[5];   // P[1,1] = focal_y
    
    // For point (x,y,z) in view space, clip coords are (fx*x, fy*y, ...) before division by w
    // Jacobian J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    // Use absolute value to ensure consistent coordinate system handling
    float inv_z = 1.0f / fabsf(vz);  // Use absolute value for robustness
    float inv_z2 = inv_z * inv_z;
    
    // Jacobian matrix elements with projection matrix scaling
    // J = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]
    float j00 = fx * inv_z;
    float j02 = -fx * vx * inv_z2;  // Added missing negative sign
    float j11 = fy * inv_z;
    float j12 = -fy * vy * inv_z2;  // Added missing negative sign
    
    // Compute 2D covariance: Σ_2D = J * Σ_view * J^T
    // J = [[j00, 0, j02], [0, j11, j12]]
    // Σ_view = [[cov_view_00, cov_view_01, cov_view_02],
    //           [cov_view_01, cov_view_11, cov_view_12],
    //           [cov_view_02, cov_view_12, cov_view_22]]
    float cov2d_00 = j00*j00*cov_view_00 + j02*j02*cov_view_22 + 2.0f*j00*j02*cov_view_02;
    float cov2d_01 = j00*j11*cov_view_01 + j02*j12*cov_view_22 + j00*j12*cov_view_02 + j02*j11*cov_view_01;
    float cov2d_11 = j11*j11*cov_view_11 + j12*j12*cov_view_22 + 2.0f*j11*j12*cov_view_12;
    
    
    // Add a much smaller regularization term to ensure positive definiteness
    // The original 1e-4f was too large and dominating the actual values
    cov2d_00 += 1e-8f;
    cov2d_11 += 1e-8f;
    
    // Store 2D covariance (symmetric, so store upper triangle)
    cov2d_data[cov_offset + 0] = cov2d_00;
    cov2d_data[cov_offset + 1] = cov2d_01;
    cov2d_data[cov_offset + 2] = cov2d_11;
    
    // Compute eigenvalues for quad sizing (3σ bounds)
    // For 2x2 symmetric matrix [[a,b],[b,c]], eigenvalues are:
    // λ = (a+c ± sqrt((a-c)² + 4b²)) / 2
    double trace = (double)cov2d_00 + (double)cov2d_11;
    double det = (double)cov2d_00 * (double)cov2d_11 - (double)cov2d_01 * (double)cov2d_01;
    double discriminant = trace * trace - 4.0 * det;
    
    if (discriminant < 0.0) {
        // Degenerate case, but still render with minimal size
        discriminant = 0.0;
    }
    
    double sqrt_disc = sqrt(discriminant);
    float lambda1 = (float)(0.5 * (trace + sqrt_disc));
    float lambda2 = (float)(0.5 * (trace - sqrt_disc));
    
    // Compute radii (3σ = 3 * sqrt(eigenvalue))
    // Note: eigenvalues are now in NDC space due to projection matrix scaling in Jacobian
    float radius_x = 3.0f * sqrtf(fmaxf(lambda1, 1e-6f));
    float radius_y = 3.0f * sqrtf(fmaxf(lambda2, 1e-6f));
    
    
    // Don't cull small quads - render everything
    // Ensure minimum size for degenerate cases
    radius_x = fmaxf(radius_x, 1e-6f);
    radius_y = fmaxf(radius_y, 1e-6f);
    
    // Project center to normalized device coordinates (NDC) using full projection
    float ndc_x, ndc_y, ndc_z;
    project_to_ndc(vx, vy, vz, vw, proj_matrix, ndc_x, ndc_y, ndc_z);
    
    // Handle invalid values by clamping
    if (!isfinite(ndc_x)) ndc_x = 0.0f;
    if (!isfinite(ndc_y)) ndc_y = 0.0f;
    if (!isfinite(ndc_z)) ndc_z = 0.0f;
    
    // The radii are already in the correct space due to projection matrix scaling in Jacobian
    // No additional conversion needed
    float radius_x_ndc = radius_x;
    float radius_y_ndc = radius_y;
    
    
    // No artificial capping needed - let the natural 3σ bounds determine the size
    // The radii are already computed from the eigenvalues with 3σ scaling
    
    // Store quad parameters (all in NDC space)
    quad_params[quad_offset + 0] = ndc_x;
    quad_params[quad_offset + 1] = ndc_y;
    quad_params[quad_offset + 2] = radius_x_ndc;
    quad_params[quad_offset + 3] = radius_y_ndc;
    quad_params[quad_offset + 4] = ndc_z;  // Store NDC z for depth
    
    // All quads are visible (already set above)
}

// Compact visible quads using prefix sum - maintains sorted order
extern "C" __global__
void compact_visible_quads(const float* quad_vertices_in,       // Input quad vertices (all quads)
                          const float* quad_uvs_in,            // Input UV coordinates (all quads)  
                          const float* quad_data_in,           // Input quad data (all quads)
                          const int* visibility_mask,         // Visibility mask
                          const int* prefix_sum,               // Prefix sum of visibility mask
                          float* quad_vertices_out,            // Output: compacted quad vertices
                          float* quad_uvs_out,                 // Output: compacted UV coordinates
                          float* quad_data_out,                // Output: compacted quad data
                          int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Skip if not visible
    if (visibility_mask[idx] == 0) return;
    
    // Get output index from prefix sum (maintains sorted order)
    int output_idx = prefix_sum[idx];
    
    // Copy quad vertices (4 vertices per quad)
    for (int v = 0; v < 4; v++) {
        int input_vertex_idx = idx * 4 + v;
        int output_vertex_idx = output_idx * 4 + v;
        
        // Copy vertex data (8 floats per vertex: x,y,z,r,g,b,center_x,center_y)
        for (int f = 0; f < 8; f++) {
            quad_vertices_out[output_vertex_idx * 8 + f] = quad_vertices_in[input_vertex_idx * 8 + f];
        }
        
        // Copy UV data (2 floats per vertex: u,v)
        for (int f = 0; f < 2; f++) {
            quad_uvs_out[output_vertex_idx * 2 + f] = quad_uvs_in[input_vertex_idx * 2 + f];
        }
    }
    
    // Copy quad data (6 floats per quad: opacity, inv_cov components, radii)
    for (int f = 0; f < 6; f++) {
        quad_data_out[output_idx * 6 + f] = quad_data_in[idx * 6 + f];
    }
}

// Generate quad vertices for visible Gaussians
extern "C" __global__
void generate_quad_vertices(const float* quad_params,           // Quad parameters (center_x, center_y, radius_x, radius_y, ndc_z)
                           const float* cov2d_data,             // 2D covariance matrices (3 components each)
                           const int* visibility_mask,         // Visibility mask
                           const float* colors,                // Colors (3 components each)
                           const float* opacities,             // Opacity values
                           float* quad_vertices,               // Output: Quad vertices (8 floats per vertex: x,y,z,r,g,b,center_x,center_y)
                           float* quad_uvs,                    // Output: UV coordinates (2 floats per vertex)
                           float* quad_data,                   // Output: Per-quad data (opacity + 2D covariance inverse)
                           int* visible_count,                 // Output: Number of visible quads
                           int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Load quad parameters - no visibility check for performance
    int param_offset = idx * 5;
    float center_x = quad_params[param_offset + 0];
    float center_y = quad_params[param_offset + 1];
    float radius_x = quad_params[param_offset + 2];
    float radius_y = quad_params[param_offset + 3];
    float ndc_z = quad_params[param_offset + 4];
    
    // Use the original sorted index to maintain depth order
    // Don't use atomic counter as it breaks sorting
    int quad_idx = idx;
    
    
    // Load 2D covariance matrix
    int cov_offset = idx * 3;
    float cov_00 = cov2d_data[cov_offset + 0];
    float cov_01 = cov2d_data[cov_offset + 1];
    float cov_11 = cov2d_data[cov_offset + 2];
    
    // Compute inverse of 2D covariance matrix for fragment shader
    float det = cov_00 * cov_11 - cov_01 * cov_01;
    
    
    // Skip if covariance matrix is degenerate (made more permissive)
    if (det <= 1e-12f || cov_00 <= 1e-12f || cov_11 <= 1e-12f) {
        return;
    }
    
    float inv_det = 1.0f / det;
    float inv_cov_00 = cov_11 * inv_det;
    float inv_cov_01 = -cov_01 * inv_det;
    float inv_cov_11 = cov_00 * inv_det;
    
    
    // Load color and opacity
    int color_offset = idx * 3;
    float r = colors[color_offset + 0];
    float g = colors[color_offset + 1];
    float b = colors[color_offset + 2];
    float opacity = opacities[idx];
    
    // Compute eigenvectors for oriented quad
    // For symmetric matrix [[a,b],[b,c]], eigenvector of larger eigenvalue:
    float trace = cov_00 + cov_11;
    float discriminant = trace * trace - 4.0f * det;
    float sqrt_disc = sqrtf(fmaxf(discriminant, 0.0f));
    float lambda1 = 0.5f * (trace + sqrt_disc);
    
    // Eigenvector corresponding to lambda1
    float evec_x, evec_y;
    if (fabsf(cov_01) > 1e-6f) {
        evec_x = lambda1 - cov_11;
        evec_y = cov_01;
        float norm = sqrtf(evec_x * evec_x + evec_y * evec_y);
        evec_x /= norm;
        evec_y /= norm;
    } else {
        // Diagonal matrix case
        evec_x = 1.0f;
        evec_y = 0.0f;
    }
    
    // Generate 4 vertices for the quad
    // Vertex layout: bottom-left, bottom-right, top-left, top-right
    float offsets_x[4] = {-1.0f, 1.0f, -1.0f, 1.0f};
    float offsets_y[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
    float uvs[8] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    
    for (int i = 0; i < 4; i++) {
        int vertex_idx = quad_idx * 4 + i;
        int vertex_offset = vertex_idx * 8;  // 8 floats per vertex (x,y,z,r,g,b,center_x,center_y)
        int uv_offset = vertex_idx * 2;      // 2 floats per vertex (u,v)
        
        // Rotate and scale offset by covariance eigenvectors
        float local_x = offsets_x[i] * radius_x;
        float local_y = offsets_y[i] * radius_y;
        
        float world_x = evec_x * local_x - evec_y * local_y;
        float world_y = evec_y * local_x + evec_x * local_y;
        
        // Store vertex position (in NDC space with proper depth)
        quad_vertices[vertex_offset + 0] = center_x + world_x;
        quad_vertices[vertex_offset + 1] = center_y + world_y;
        quad_vertices[vertex_offset + 2] = ndc_z;  // Use the NDC z-coordinate for proper depth
        
        // Store vertex color normally
        quad_vertices[vertex_offset + 3] = r;
        quad_vertices[vertex_offset + 4] = g;
        quad_vertices[vertex_offset + 5] = b;
        
        // Store Gaussian center position in NDC space (new!)
        quad_vertices[vertex_offset + 6] = center_x;
        quad_vertices[vertex_offset + 7] = center_y;
        
        // Store UV coordinates
        quad_uvs[uv_offset + 0] = uvs[i * 2 + 0];
        quad_uvs[uv_offset + 1] = uvs[i * 2 + 1];
    }
    
    // Store per-quad data for fragment shader
    // Extend to 6 components: opacity, inv_cov (3), radii (2)
    int quad_data_offset = quad_idx * 6;
    quad_data[quad_data_offset + 0] = opacity;
    quad_data[quad_data_offset + 1] = inv_cov_00;
    quad_data[quad_data_offset + 2] = inv_cov_01;
    quad_data[quad_data_offset + 3] = inv_cov_11;
    quad_data[quad_data_offset + 4] = radius_x;  // NDC radius in X direction
    quad_data[quad_data_offset + 5] = radius_y;  // NDC radius in Y direction
}

// Generate indices for rendering quads as triangles
extern "C" __global__
void generate_quad_indices(unsigned int* indices,    // Output: Triangle indices
                          int num_quads) {           // Number of quads to generate indices for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_quads) return;
    
    // Each quad needs 6 indices (2 triangles)
    // Quad vertices are ordered: bottom-left(0), bottom-right(1), top-left(2), top-right(3)
    int base_vertex = idx * 4;
    int base_index = idx * 6;
    
    // First triangle: bottom-left, bottom-right, top-left (0, 1, 2)
    indices[base_index + 0] = base_vertex + 0;
    indices[base_index + 1] = base_vertex + 1;
    indices[base_index + 2] = base_vertex + 2;
    
    // Second triangle: bottom-right, top-right, top-left (1, 3, 2)
    indices[base_index + 3] = base_vertex + 1;
    indices[base_index + 4] = base_vertex + 3;
    indices[base_index + 5] = base_vertex + 2;
}

// Generate instance data for instanced rendering
// Each instance represents one Gaussian with all its properties
extern "C" __global__
void generate_instance_data(const float* quad_params,     // Quad parameters (center_x, center_y, radius_x, radius_y, ndc_z)
                           const float* cov2d_data,       // 2D covariance matrices (3 components each)
                           const int* visibility_mask,    // Visibility mask
                           const float* colors,           // Colors (3 components each)
                           const float* opacities,        // Opacity values
                           float* instance_data,          // Output: Instance data (10 floats per instance)
                           int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    // Load quad parameters - no visibility check
    int param_offset = idx * 5;
    float center_x = quad_params[param_offset + 0];
    float center_y = quad_params[param_offset + 1];
    float radius_x = quad_params[param_offset + 2];
    float radius_y = quad_params[param_offset + 3];
    float ndc_z = quad_params[param_offset + 4];
    
    // Load 2D covariance matrix
    int cov_offset = idx * 3;
    float cov_00 = cov2d_data[cov_offset + 0];
    float cov_01 = cov2d_data[cov_offset + 1];
    float cov_11 = cov2d_data[cov_offset + 2];
    
    // Compute inverse of 2D covariance matrix
    float det = cov_00 * cov_11 - cov_01 * cov_01;
    if (det <= 1e-12f) det = 1e-12f;  // Ensure non-zero determinant
    
    float inv_det = 1.0f / det;
    float inv_cov_00 = cov_11 * inv_det;
    float inv_cov_01 = -cov_01 * inv_det;
    float inv_cov_11 = cov_00 * inv_det;
    
    // Load color and opacity
    int color_offset = idx * 3;
    float r = colors[color_offset + 0];
    float g = colors[color_offset + 1];
    float b = colors[color_offset + 2];
    float opacity = opacities[idx];
    
    // Pack instance data (10 floats per instance)
    // Layout: center_x, center_y, ndc_z, r, g, b, opacity, inv_cov_00, inv_cov_01, inv_cov_11
    int instance_offset = idx * 10;
    instance_data[instance_offset + 0] = center_x;
    instance_data[instance_offset + 1] = center_y;
    instance_data[instance_offset + 2] = ndc_z;
    instance_data[instance_offset + 3] = r;
    instance_data[instance_offset + 4] = g;
    instance_data[instance_offset + 5] = b;
    instance_data[instance_offset + 6] = opacity;
    instance_data[instance_offset + 7] = inv_cov_00;
    instance_data[instance_offset + 8] = inv_cov_01;
    instance_data[instance_offset + 9] = inv_cov_11;
}