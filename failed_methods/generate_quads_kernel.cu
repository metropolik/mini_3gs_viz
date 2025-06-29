
extern "C" __global__
void generate_quads(
    const int* sorted_indices,   // (N,) sorted indices
    const float* screen_pos,     // (N, 2) screen positions
    const float* cov2d,          // (N, 3) 2D covariances
    const float* colors,         // (N, 3) RGB colors
    const float* opacities,      // (N,) opacities
    float* vertices,            // (N*4, 3) output vertex positions (x,y,z)
    float* vertex_colors,       // (N*4, 4) output vertex colors (r,g,b,a)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Use sorted index to determine which splat to render at this position
    int i = sorted_indices[idx];
    
    // Get 2D covariance parameters
    float a = cov2d[i * 3 + 0];
    float b = cov2d[i * 3 + 1];
    float c = cov2d[i * 3 + 2];
    
    // Screen position
    float sx = screen_pos[i * 2 + 0];
    float sy = screen_pos[i * 2 + 1];
    
    // Calculate eigenvalues for ellipse axes (3-sigma)
    float det = a * c - b * b;
    
    float radius_x = 0.0f;
    float radius_y = 0.0f;
    float alpha = opacities[i];
    
    // Check if valid and on screen
    if (det > 0 && fabsf(sx) <= 1.5f && fabsf(sy) <= 1.5f) {
        float trace = a + c;
        float sqrt_discriminant = sqrtf(trace * trace - 4 * det);
        float lambda1 = 0.5f * (trace + sqrt_discriminant);
        float lambda2 = 0.5f * (trace - sqrt_discriminant);
        
        // 3-sigma radii with scale factor
        float scale_factor = 0.05f;  // Adjust this to control splat size
        radius_x = 3.0f * sqrtf(lambda1) * scale_factor;
        radius_y = 3.0f * sqrtf(lambda2) * scale_factor;
    } else {
        // Make invalid quads invisible
        alpha = 0.0f;
    }
    
    // Generate quad vertices (counter-clockwise)
    int base_idx = idx * 4;
    
    // Bottom-left
    vertices[(base_idx + 0) * 3 + 0] = sx - radius_x;
    vertices[(base_idx + 0) * 3 + 1] = sy - radius_y;
    vertices[(base_idx + 0) * 3 + 2] = 0.0f;
    
    // Bottom-right
    vertices[(base_idx + 1) * 3 + 0] = sx + radius_x;
    vertices[(base_idx + 1) * 3 + 1] = sy - radius_y;
    vertices[(base_idx + 1) * 3 + 2] = 0.0f;
    
    // Top-right
    vertices[(base_idx + 2) * 3 + 0] = sx + radius_x;
    vertices[(base_idx + 2) * 3 + 1] = sy + radius_y;
    vertices[(base_idx + 2) * 3 + 2] = 0.0f;
    
    // Top-left
    vertices[(base_idx + 3) * 3 + 0] = sx - radius_x;
    vertices[(base_idx + 3) * 3 + 1] = sy + radius_y;
    vertices[(base_idx + 3) * 3 + 2] = 0.0f;
    
    // Set colors for all 4 vertices
    float red = colors[i * 3 + 0];
    float green = colors[i * 3 + 1];
    float blue = colors[i * 3 + 2];
    
    for (int v = 0; v < 4; v++) {
        vertex_colors[(base_idx + v) * 4 + 0] = red;
        vertex_colors[(base_idx + v) * 4 + 1] = green;
        vertex_colors[(base_idx + v) * 4 + 2] = blue;
        vertex_colors[(base_idx + v) * 4 + 3] = alpha;
    }
}