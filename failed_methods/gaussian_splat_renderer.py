import numpy as np
import open3d as o3d
import open3d.core as o3c
from OpenGL.GL import *
from OpenGL.GLU import *
from math import radians, sin, cos
import cupy as cp
import time

# CuPy kernel for projecting 3D Gaussians to 2D and calculating screen-space covariance
with open("project_gaussians.cu") as kernel_file:
    project_gaussians_kernel = cp.RawKernel(kernel_file.read(), 'project_gaussians')

# CUDA kernel for generating quad vertices from sorted Gaussians
#kernel_file_name = "generate_quads_kernel.cu"
kernel_file_name = "simple_generate_quads_kernel.cu" # for debugging purposes
with open(kernel_file_name) as kernel_file:
    generate_quads_kernel = cp.RawKernel(kernel_file.read(), 'generate_quads')

kernel_file_name = "move_points_kernel.cu" # for debugging purposes
with open(kernel_file_name) as kernel_file:
    move_points_kernel = cp.RawKernel(kernel_file.read(), 'move_points')

# CUDA kernel for generating random points
with open("random_points_kernel.cu") as kernel_file:
    random_points_kernel = cp.RawKernel(kernel_file.read(), 'generate_random_points')

with open("transform_points_kernel.cu") as kernel_file:
    transform_points_kernel = cp.RawKernel(kernel_file.read(), 'transform_points')

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
        
    def render(self, camera_position, view_matrix, projection_matrix, debug_mode=None):
        if debug_mode == 'gpu_test':
            # Simple GPU test: copy to GPU, add 0.5, copy back, render
            N = len(self.positions)
            print(f"\nGPU Test Mode:")
            print(f"Original positions[0]: {self.positions[0]}")
            
            # Copy to GPU
            positions_gpu = cp.asarray(self.positions, dtype=cp.float32)
            
            # Use kernel to move points
            threads_per_block = 256
            blocks = (N + threads_per_block - 1) // threads_per_block
            
            move_points_kernel(
                (blocks,), (threads_per_block,),
                (positions_gpu, cp.float32(0.5), cp.float32(0.5), cp.float32(0.5), N)
            )
            cp.cuda.runtime.deviceSynchronize()
            
            # Copy back
            positions_modified = positions_gpu.get()
            print(f"Modified positions[0]: {positions_modified[0]}")
            print(f"First 5 original positions:")
            for i in range(min(5, N)):
                print(f"  [{i}]: {self.positions[i]}")
            print(f"First 5 modified positions:")
            for i in range(min(5, N)):
                print(f"  [{i}]: {positions_modified[i]}")
            
            # Render with simple point rendering
            #glMatrixMode(GL_PROJECTION)
            #glLoadIdentity()
            #glMatrixMode(GL_MODELVIEW)
            #glLoadIdentity()
            
            glEnable(GL_POINT_SMOOTH)
            glPointSize(3.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            # Scale down to fit in NDC
            
            glVertexPointer(3, GL_FLOAT, 0, positions_modified)
            glColorPointer(3, GL_FLOAT, 0, self.colors)
            glDrawArrays(GL_POINTS, 0, N)
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_POINT_SMOOTH)
            return
        if self.positions is None:
            return
        
        N = len(self.positions)
        
        # Debug: Check for (0,0,0) points
        zero_points = np.all(self.positions == 0, axis=1)
        num_zeros = np.sum(zero_points)
        if num_zeros > 0:
            print(f"WARNING: Found {num_zeros} points at (0,0,0)")
            zero_indices = np.where(zero_points)[0]
            print(f"  Indices: {zero_indices[:10]}...")  # Show first 10
        
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
        
        # Debug: Print first few matrix elements
        print(f"View matrix (first row): [{view_matrix[0,0]:.3f}, {view_matrix[0,1]:.3f}, {view_matrix[0,2]:.3f}, {view_matrix[0,3]:.3f}]")
        print(f"Proj matrix (first row): [{projection_matrix[0,0]:.3f}, {projection_matrix[0,1]:.3f}, {projection_matrix[0,2]:.3f}, {projection_matrix[0,3]:.3f}]")
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
        
        # Debug: Check screen positions range
        screen_pos_cpu = screen_pos_gpu.get()
        print(f"Screen pos X range: [{screen_pos_cpu[:,0].min():.3f}, {screen_pos_cpu[:,0].max():.3f}]")
        print(f"Screen pos Y range: [{screen_pos_cpu[:,1].min():.3f}, {screen_pos_cpu[:,1].max():.3f}]")
        
        # Debug: Print detailed transformation info for point 100
        cov2d_cpu = cov2d_gpu.get()
            
        # Also check depths
        depths_cpu = depths_gpu.get()
        
        # Check for points at different depths
        near_idx = np.argmax(depths_cpu)  # Closest point (largest negative z)
        far_idx = np.argmin(depths_cpu)   # Farthest point
        
        # Debug modes
        if debug_mode == 'points_3d':
            # Render points using normal 3D projection
            glMatrixMode(GL_PROJECTION)
            glLoadMatrixf(projection_matrix.T.flatten())
            glMatrixMode(GL_MODELVIEW)
            glLoadMatrixf(view_matrix.T.flatten())
            
            glEnable(GL_POINT_SMOOTH)
            glPointSize(2.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.positions)
            glColorPointer(3, GL_FLOAT, 0, self.colors)
            glDrawArrays(GL_POINTS, 0, N)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_POINT_SMOOTH)
            return
            
        elif debug_mode == 'points_2d':
            # Generate random 3D points to test rendering
            print(f"\nGenerating {N} random 3D points in [-1,1] range")
            
            # Allocate GPU memory for random points
            random_points_gpu = cp.zeros((N, 3), dtype=cp.float32)
            
            # Generate random points using kernel
            threads_per_block = 256
            blocks = (N + threads_per_block - 1) // threads_per_block
            seed = np.uint32(42)  # Fixed seed for reproducibility
            
            random_points_kernel(
                (blocks,), (threads_per_block,),
                (random_points_gpu, seed, N)
            )
            cp.cuda.runtime.deviceSynchronize()
            
            # Copy back to CPU
            random_points_cpu = random_points_gpu.get()
            print(f"Random points range X: [{random_points_cpu[:,0].min():.3f}, {random_points_cpu[:,0].max():.3f}]")
            print(f"Random points range Y: [{random_points_cpu[:,1].min():.3f}, {random_points_cpu[:,1].max():.3f}]")
            print(f"Random points range Z: [{random_points_cpu[:,2].min():.3f}, {random_points_cpu[:,2].max():.3f}]")
            
            # Render points directly without any matrix transformations
            glEnable(GL_POINT_SMOOTH)
            glPointSize(2.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, random_points_cpu)
            glColorPointer(3, GL_FLOAT, 0, self.colors)
            glDrawArrays(GL_POINTS, 0, N)
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_POINT_SMOOTH)
            return
            
        elif debug_mode == 'transform_test':
            # Test our own modelview transformation
            print(f"\nTesting custom modelview transformation")
            
            # Test with a single known point first
            test_point = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            print(f"\nTest point: {test_point[0]}")
            
            # Apply OpenGL transformation manually in numpy
            test_homogeneous = np.append(test_point[0], 1.0)  # Add w=1
            transformed_opengl = view_matrix @ test_homogeneous
            print(f"NumPy matrix multiply result: {transformed_opengl[:3]}")
            
            # Now test our kernel
            # Copy positions to GPU
            positions_gpu = cp.asarray(self.positions, dtype=cp.float32)
            transformed_gpu = cp.zeros_like(positions_gpu)
            
            # Debug: print view matrix
            print(f"View matrix:")
            for i in range(4):
                print(f"  [{view_matrix[i,0]:.3f}, {view_matrix[i,1]:.3f}, {view_matrix[i,2]:.3f}, {view_matrix[i,3]:.3f}]")
            
            # Try inverting the view matrix (camera transform vs view transform)
            # view_inverted = np.linalg.inv(view_matrix)
            # print(f"\nInverted view matrix:")
            # for i in range(4):
            #     print(f"  [{view_inverted[i,0]:.3f}, {view_inverted[i,1]:.3f}, {view_inverted[i,2]:.3f}, {view_inverted[i,3]:.3f}]")
            
            # Flatten view matrix for GPU
            view_flat = cp.asarray(view_matrix.flatten(), dtype=cp.float32)
            print(f"Flattened (first 16): {view_flat.get()}")
            
            # Apply transformation using kernel
            threads_per_block = 256
            blocks = (N + threads_per_block - 1) // threads_per_block
            
            transform_points_kernel(
                (blocks,), (threads_per_block,),
                (positions_gpu, view_flat, transformed_gpu, N)
            )
            cp.cuda.runtime.deviceSynchronize()
            
            # Copy back
            transformed_positions = transformed_gpu.get()
            
            # Print comparison for first few points
            print(f"First 3 original positions:")
            for i in range(min(3, N)):
                print(f"  [{i}]: {self.positions[i]}")
            print(f"First 3 transformed positions:")
            for i in range(min(3, N)):
                print(f"  [{i}]: {transformed_positions[i]}")
            
            # Don't touch projection matrix - just set modelview to identity
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()  # Identity since we already transformed
            
            glEnable(GL_POINT_SMOOTH)
            glPointSize(2.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, transformed_positions)
            glColorPointer(3, GL_FLOAT, 0, self.colors)
            glDrawArrays(GL_POINTS, 0, N)
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_POINT_SMOOTH)
            return
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_POINT_SMOOTH)
            
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            return
        
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
        
        # Disable depth test for proper transparency
        glDisable(GL_DEPTH_TEST)
        
        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity() # coordinates are now normalized device coords [-1,1]^3
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
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
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)  # Re-enable depth test
        print("---")