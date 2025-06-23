import numpy as np
import open3d as o3d
import open3d.core as o3c
from OpenGL.GL import *
from OpenGL.GLU import *
from math import radians, sin, cos
import cupy as cp
import time

# CuPy kernel for projecting 3D Gaussians to 2D and calculating screen-space covariance
project_gaussians_kernel = cp.RawKernel(r'''
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
    
    depths[idx] = -vz;  // Negative for proper sorting (closer = larger)
    
    // Project to screen space
    float px = proj_matrix[0] * vx + proj_matrix[4] * vy + proj_matrix[8] * vz + proj_matrix[12];
    float py = proj_matrix[1] * vx + proj_matrix[5] * vy + proj_matrix[9] * vz + proj_matrix[13];
    float pw = proj_matrix[3] * vx + proj_matrix[7] * vy + proj_matrix[11] * vz + proj_matrix[15];
    
    screen_pos[idx * 2 + 0] = px / pw;
    screen_pos[idx * 2 + 1] = py / pw;
    
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
''', 'project_gaussians')

# CUDA kernel for generating quad vertices from sorted Gaussians
generate_quads_kernel = cp.RawKernel(r'''
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
        
        // 3-sigma radii
        radius_x = 3.0f * sqrtf(lambda1);
        radius_y = 3.0f * sqrtf(lambda2);
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
''', 'generate_quads')

class GaussianSplatRenderer:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.positions = None
        self.colors = None
        self.scales = None
        self.rotations = None
        self.opacities = None
        self.sh_coeffs = None
        
        # VBO handles
        self.vbo_vertices = None
        self.vbo_colors = None
        self.vertices_gpu = None
        self.vertex_colors_gpu = None
        
        self.load_ply()
        
    def load_ply(self):
        print(f"Loading Gaussian splat model from {self.ply_path}")
        
        # Load PLY file using Open3D tensor API
        pcd_t = o3d.t.io.read_point_cloud(self.ply_path)
        
        # Print all available point attributes
        print("\nAvailable point attributes:")
        for attr_name in dir(pcd_t.point):
            if not attr_name.startswith('_'):
                attr = getattr(pcd_t.point, attr_name)
                if hasattr(attr, 'shape') and hasattr(attr, 'dtype'):
                    print(f"  {attr_name}: shape={attr.shape}, dtype={attr.dtype}")
        
        # Extract positions (x, y, z)
        if pcd_t.point.positions is not None:
            self.positions = pcd_t.point.positions.numpy()
            rot_angle = radians(180)
            rots = sin(rot_angle)
            rotc = cos(rot_angle)
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, rotc, -rots],
                                [0.0, rots, rotc]])
            # Apply rotation to all points: (3x3) @ (3xN) = (3xN), then transpose back
            self.positions = (rot_mat @ self.positions.T).T
            print(f"\nLoaded {len(self.positions)} splats")
        
        # Extract colors if available (f_dc_0, f_dc_1, f_dc_2)
        if 'f_dc_0' in pcd_t.point:
            print("Found colors")
            dc_0 = pcd_t.point['f_dc_0'].numpy()
            dc_1 = pcd_t.point['f_dc_1'].numpy()
            dc_2 = pcd_t.point['f_dc_2'].numpy()
            self.colors = np.stack([dc_0, dc_1, dc_2], axis=-1)
            # Squeeze out any extra dimensions
            self.colors = np.squeeze(self.colors)
            # Ensure correct shape (num_points, 3)
            if self.colors.ndim == 3 and self.colors.shape[1] == 1:
                self.colors = self.colors.squeeze(1)
            # Convert from SH DC term to RGB (SH DC term is scaled by 0.28209479177387814)
            self.colors = 0.5 + self.colors * 0.28209479177387814
            self.colors = np.clip(self.colors, 0, 1)
            print(f"Colors shape {self.colors.shape}")
        else:
            print("did not find colors, using white instead")
            self.colors = np.ones((len(self.positions), 3))
        
        # Extract scales (scale_0, scale_1, scale_2)
        if 'scale_0' in pcd_t.point:
            scale_0 = pcd_t.point['scale_0'].numpy()
            scale_1 = pcd_t.point['scale_1'].numpy()
            scale_2 = pcd_t.point['scale_2'].numpy()
            self.scales = np.exp(np.stack([scale_0, scale_1, scale_2], axis=-1))
        
        # Extract rotations (rot_0, rot_1, rot_2, rot_3) - quaternions
        if 'rot_0' in pcd_t.point:
            rot_0 = pcd_t.point['rot_0'].numpy()
            rot_1 = pcd_t.point['rot_1'].numpy()
            rot_2 = pcd_t.point['rot_2'].numpy()
            rot_3 = pcd_t.point['rot_3'].numpy()
            self.rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1)
            # Normalize quaternions
            norms = np.linalg.norm(self.rotations, axis=-1, keepdims=True)
            self.rotations = self.rotations / (norms + 1e-8)
        
        # Extract opacity
        if 'opacity' in pcd_t.point:
            self.opacities = pcd_t.point['opacity'].numpy()
            # Apply sigmoid activation
            self.opacities = 1 / (1 + np.exp(-self.opacities))
        
        # Extract spherical harmonics coefficients if available
        sh_features = []
        sh_degree = 0
        for degree in range(4):  # Check up to degree 3
            if degree == 0:
                # Already loaded as colors (f_dc_0, f_dc_1, f_dc_2)
                continue
            else:
                # Check if SH coefficients exist for this degree
                sh_exists = True
                for l in range(degree * 2 + 1):
                    for c in range(3):  # RGB channels
                        feature_name = f'f_rest_{(degree-1)*9 + l*3 + c}'
                        if feature_name not in pcd_t.point:
                            sh_exists = False
                            break
                    if not sh_exists:
                        break
                
                if sh_exists:
                    sh_degree = degree
                    # Load SH coefficients for this degree
                    for l in range(degree * 2 + 1):
                        for c in range(3):
                            feature_name = f'f_rest_{(degree-1)*9 + l*3 + c}'
                            sh_features.append(pcd_t.point[feature_name].numpy())
        
        if sh_features:
            self.sh_coeffs = np.stack(sh_features, axis=-1)
            print(f"Loaded spherical harmonics up to degree {sh_degree}")
        
    def render(self, camera_position, view_matrix, projection_matrix):
        if self.positions is None:
            return
        
        N = len(self.positions)
        
        # Convert data to CuPy arrays
        t0 = time.perf_counter()
        positions_gpu = cp.asarray(self.positions, dtype=cp.float32)
        scales_gpu = cp.asarray(self.scales if self.scales is not None else np.ones((N, 3)), dtype=cp.float32)
        rotations_gpu = cp.asarray(self.rotations if self.rotations is not None else np.tile([1, 0, 0, 0], (N, 1)), dtype=cp.float32)
        colors_gpu = cp.asarray(self.colors, dtype=cp.float32)
        opacities_gpu = cp.asarray(self.opacities if self.opacities is not None else np.ones(N) * 0.9, dtype=cp.float32)
        
        # Flatten matrices for GPU
        view_flat = cp.asarray(view_matrix.flatten(), dtype=cp.float32)
        proj_flat = cp.asarray(projection_matrix.flatten(), dtype=cp.float32)
        t1 = time.perf_counter()
        print(f"GPU upload: {(t1-t0)*1000:.2f}ms")
        
        # Allocate output arrays for projection
        t0 = time.perf_counter()
        depths_gpu = cp.zeros(N, dtype=cp.float32)
        cov2d_gpu = cp.zeros((N, 3), dtype=cp.float32)
        screen_pos_gpu = cp.zeros((N, 2), dtype=cp.float32)
        t1 = time.perf_counter()
        print(f"GPU allocation: {(t1-t0)*1000:.2f}ms")
        
        # Launch projection kernel
        t0 = time.perf_counter()
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        project_gaussians_kernel(
            (blocks,), (threads_per_block,),
            (positions_gpu, scales_gpu, rotations_gpu, view_flat, proj_flat,
             depths_gpu, cov2d_gpu, screen_pos_gpu, N)
        )
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()
        print(f"Projection kernel: {(t1-t0)*1000:.2f}ms")
        
        # Sort by depth
        t0 = time.perf_counter()
        sort_indices = cp.argsort(depths_gpu)
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()
        print(f"GPU sort: {(t1-t0)*1000:.2f}ms")
        
        # Allocate output arrays for quad generation
        t0 = time.perf_counter()
        vertices_gpu = cp.zeros((N * 4, 3), dtype=cp.float32)
        vertex_colors_gpu = cp.zeros((N * 4, 4), dtype=cp.float32)
        t1 = time.perf_counter()
        print(f"Quad GPU allocation: {(t1-t0)*1000:.2f}ms")
        
        # Launch quad generation kernel
        t0 = time.perf_counter()
        generate_quads_kernel(
            (blocks,), (threads_per_block,),
            (sort_indices, screen_pos_gpu, cov2d_gpu, colors_gpu, opacities_gpu,
             vertices_gpu, vertex_colors_gpu, N)
        )
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()
        print(f"Quad generation kernel: {(t1-t0)*1000:.2f}ms")
        
        # Transfer data back to CPU
        t0 = time.perf_counter()
        vertices = vertices_gpu.get()
        vertex_colors = vertex_colors_gpu.get()
        t1 = time.perf_counter()
        print(f"GPU to CPU transfer: {(t1-t0)*1000:.2f}ms")
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glColorPointer(4, GL_FLOAT, 0, vertex_colors)
        
        # Draw all quads directly
        t0 = time.perf_counter()
        # Draw using GL_QUADS (each 4 vertices forms a quad)
        glDrawArrays(GL_QUADS, 0, N * 4)
        t1 = time.perf_counter()
        print(f"OpenGL draw: {(t1-t0)*1000:.2f}ms")
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisable(GL_BLEND)
        print("---")