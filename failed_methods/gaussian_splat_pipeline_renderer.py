import numpy as np
import open3d as o3d
import open3d.core as o3c
import cupy as cp
from OpenGL.GL import *
from OpenGL.GLU import *
from math import radians, sin, cos

class GaussianSplatRenderer:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.positions = None
        self.colors = None
        self.scales = None
        self.rotations = None
        self.opacities = None
        self.sh_coeffs = None
        
        # CUDA kernel for transformations
        self.transform_kernel = None
        self.load_transform_kernel()
        
        # GPU buffers
        self.positions_gpu = None
        self.transformed_positions_gpu = None
        
        self.load_ply()
        
    def load_transform_kernel(self):
        """Load and compile the CUDA kernel for point transformations"""
        with open('transform_kernel.cu', 'r') as f:
            kernel_code = f.read()
        
        self.transform_kernel = cp.RawKernel(kernel_code, 'transform_points')
        print("Loaded and compiled transform kernel")
        
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
        
        # Upload positions to GPU
        if self.positions is not None:
            self.positions_gpu = cp.asarray(self.positions, dtype=cp.float32)
            self.transformed_positions_gpu = cp.empty_like(self.positions_gpu)
            print(f"Uploaded positions to GPU")
        
    def render(self, camera_position, view_matrix, projection_matrix):
        if self.positions is None:
            return
        
        # Temporary: use original positions to debug
        USE_CUDA_TRANSFORM = True
        
        if USE_CUDA_TRANSFORM:
            # Compute MVP matrix: projection * view
            projection_matrix_f32 = projection_matrix.astype(np.float32)
            view_matrix_f32 = view_matrix.astype(np.float32)
            mvp_matrix = projection_matrix_f32 @ view_matrix_f32
            
            # Debug: Print the view matrix we're using
            print(f"View matrix (printed in row-major format):")
            print(view_matrix_f32)
            print(f"Camera yaw: {camera_position}, rotation part:")
            print(view_matrix_f32[:3, :3])
            
            # Debug: Let's check a test point transformation
            test_point = np.array([0, 0, 0, 1])  # Origin
            transformed_test = view_matrix_f32 @ test_point
            print(f"Test point [0,0,0] -> {transformed_test[:3]}")
            print(f"Camera position: {camera_position}")
            
            # Convert MVP matrix to GPU
            transform_gpu = cp.asarray(mvp_matrix, dtype=cp.float32)
            
            # Configure kernel launch parameters
            num_points = len(self.positions)
            threads_per_block = 256
            blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
            
            # Launch kernel
            self.transform_kernel((blocks_per_grid,), (threads_per_block,),
                                (self.positions_gpu, self.transformed_positions_gpu,
                                 transform_gpu, num_points))
            
            # Copy transformed positions back to CPU for OpenGL
            transformed_positions_cpu = cp.asnumpy(self.transformed_positions_gpu)
            
            positions_to_render = transformed_positions_cpu
            
            # Set both MODELVIEW and PROJECTION to identity (transformations done in CUDA)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

        else:
            # Use original positions but apply the given view_matrix to GL_MODELVIEW
            positions_to_render = self.positions
        
        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # Set point size
        glPointSize(3.0)
        
        # Set vertex data
        glVertexPointer(3, GL_FLOAT, 0, positions_to_render)
        
        # Set color data
        if self.colors is not None:
            glColorPointer(3, GL_FLOAT, 0, self.colors)
        else:
            # Use white color for all points
            white_colors = np.ones((len(self.positions), 3), dtype=np.float32)
            glColorPointer(3, GL_FLOAT, 0, white_colors)
        
        # Draw all points at once
        glDrawArrays(GL_POINTS, 0, len(self.positions))
        
        # Disable vertex arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        if USE_CUDA_TRANSFORM:
            # Restore OpenGL matrices
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
        
        # TODO: Implement proper Gaussian splat rendering with sorting, splatting, etc.