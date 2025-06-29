
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
    
    // Screen position
    float sx = screen_pos[i * 2 + 0];
    float sy = screen_pos[i * 2 + 1];
    
    // Use a reasonable radius for NDC coordinates (range -1 to 1)
    // 0.05 means 5% of half-screen width
    float radius_x = 0.05f;
    float radius_y = 0.05f;
    float alpha = opacities[i];
    
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