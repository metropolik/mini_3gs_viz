from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians
import os
import open3d as o3d
import ctypes
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, falling back to CPU transformations")

class PointRenderer:
    def __init__(self, ply_path=None):
        """Initialize point renderer with optional PLY file"""
        self.ply_path = ply_path
        
        # Shader source code - now expects pre-transformed vertices
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        
        out vec3 color;
        
        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            gl_PointSize = 10.0;
            color = aColor;
        }
        """
        
        self.fragment_shader_source = """
        #version 330 core
        in vec3 color;
        out vec4 FragColor;
        
        void main()
        {
            FragColor = vec4(color, 1.0);
        }
        """
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.num_points = 0
        
        # CPU transformation objects
        self.original_positions = None  # Original positions (CPU)
        self.colors = None              # Colors (CPU)
        self.scales = None              # Scale vectors (CPU)
        self.rotations = None           # Rotation quaternions (CPU)
        self.opacity = None             # Opacity values (CPU)
        self.use_cuda = False  # Disable CUDA for now
        self._debug_printed = False  # Debug flag
        
        # CUDA kernels for transformations
        self.transform_kernel = None
        self.transform_perspective_kernel = None
        self.covariance_kernel = None
        if CUPY_AVAILABLE:
            self._load_transform_kernel()
        else:
            raise RuntimeError("CuPy is required for the custom transform kernel")
        
        # Initialize if PLY file is provided
        if ply_path and os.path.exists(ply_path):
            self._setup_shaders()
            self._load_ply()
            # Enable point size control
            glEnable(GL_PROGRAM_POINT_SIZE)
        elif ply_path:
            print(f"PLY file '{ply_path}' not found, point rendering disabled")
    
    def _compile_shader(self, source, shader_type):
        """Compile a shader from source"""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation failed: {error}")
        
        return shader
    
    def _setup_shaders(self):
        """Create and compile shaders"""
        print("Setting up point shaders...")
        
        # Compile shaders
        vertex_shader = self._compile_shader(self.vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(self.fragment_shader_source, GL_FRAGMENT_SHADER)
        
        # Create shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Program linking failed: {error}")
        
        # Clean up shaders
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        # No uniforms needed for pre-transformed vertices
    
    def _load_transform_kernel(self):
        """Load the custom CUDA transform kernels"""
        try:
            with open('transform_kernel.cu', 'r') as f:
                kernel_source = f.read()
            self.transform_kernel = cp.RawKernel(kernel_source, 'transform_points')
            self.transform_perspective_kernel = cp.RawKernel(kernel_source, 'transform_points_with_perspective')
            self.covariance_kernel = cp.RawKernel(kernel_source, 'compute_2d_covariance')
            print("Custom transform and covariance kernels loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load transform kernels: {e}")
    
    
    def _load_ply(self):
        """Load PLY file and extract positions and colors"""
        print(f"Loading Gaussian splat model from {self.ply_path}")
        
        # Load PLY file using Open3D tensor API
        pcd_t = o3d.t.io.read_point_cloud(self.ply_path)
        
        # Debug: Print all available fields in the PLY file
        print("=== PLY FILE STRUCTURE DEBUG ===")
        print("Available point fields:")
        sh_fields = []
        
        # Get available field names by checking what's accessible
        available_fields = []
        test_fields = ['positions', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'scale_0', 'scale_1', 'scale_2', 
                      'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity']
        
        for field_name in test_fields:
            try:
                if hasattr(pcd_t.point, field_name):
                    field_data = getattr(pcd_t.point, field_name)
                    available_fields.append(field_name)
                    print(f"  {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")
                    # Check for spherical harmonics fields
                    if field_name.startswith('f_dc_') or field_name.startswith('f_rest_'):
                        sh_fields.append(field_name)
                elif field_name in pcd_t.point:
                    field_data = pcd_t.point[field_name]
                    available_fields.append(field_name)
                    print(f"  {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")
                    # Check for spherical harmonics fields
                    if field_name.startswith('f_dc_') or field_name.startswith('f_rest_'):
                        sh_fields.append(field_name)
            except:
                continue
        
        # Try to get additional field names that might exist
        try:
            # Some versions might have a different way to access field names
            if hasattr(pcd_t.point, 'get_dtype'):
                print("Point cloud dtype info:", pcd_t.point.get_dtype())
        except:
            pass
        
        if sh_fields:
            print(f"Found {len(sh_fields)} spherical harmonics fields: {sh_fields}")
        
        # Check for common Gaussian splatting fields
        common_fields = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity']
        missing_fields = [f for f in common_fields if f not in available_fields]
        if missing_fields:
            print(f"Missing common Gaussian splatting fields: {missing_fields}")
        else:
            print("All common Gaussian splatting fields are present!")
        
        print("=== END PLY STRUCTURE DEBUG ===")
        
        # Extract positions (x, y, z)
        positions = None
        if pcd_t.point.positions is not None:
            positions = pcd_t.point.positions.numpy()
            # Apply 180-degree rotation around X axis to match coordinate system
            rot_angle = radians(180)
            rots = sin(rot_angle)
            rotc = cos(rot_angle)
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, rotc, -rots],
                                [0.0, rots, rotc]])
            positions = (rot_mat @ positions.T).T
            print(f"Loaded {len(positions)} points")
        
        # Extract colors if available (f_dc_0, f_dc_1, f_dc_2)
        colors = None
        if 'f_dc_0' in pcd_t.point:
            print("Found spherical harmonics colors")
            dc_0 = pcd_t.point['f_dc_0'].numpy()
            dc_1 = pcd_t.point['f_dc_1'].numpy()
            dc_2 = pcd_t.point['f_dc_2'].numpy()
            colors = np.stack([dc_0, dc_1, dc_2], axis=-1)
            colors = np.squeeze(colors)
            if colors.ndim == 3 and colors.shape[1] == 1:
                colors = colors.squeeze(1)
            # Convert from SH DC term to RGB (SH DC term is scaled)
            colors = 0.5 + colors * 0.28209479177387814
            colors = np.clip(colors, 0, 1)
        else:
            print("No colors found, using white")
            colors = np.ones((len(positions), 3))
        
        # Extract scale vectors (scale_0, scale_1, scale_2)
        scales = None
        if 'scale_0' in pcd_t.point and 'scale_1' in pcd_t.point and 'scale_2' in pcd_t.point:
            print("Found scale vectors")
            scale_0 = pcd_t.point['scale_0'].numpy()
            scale_1 = pcd_t.point['scale_1'].numpy()
            scale_2 = pcd_t.point['scale_2'].numpy()
            scales = np.stack([scale_0, scale_1, scale_2], axis=-1)
            scales = np.squeeze(scales)
            if scales.ndim == 3 and scales.shape[1] == 1:
                scales = scales.squeeze(1)
            print(f"Scale data shape: {scales.shape}, range: [{scales.min():.4f}, {scales.max():.4f}]")
        else:
            print("No scale vectors found, using default scales")
            scales = np.ones((len(positions), 3)) * 0.01  # Small default scale
        
        # Extract rotation quaternions (rot_0, rot_1, rot_2, rot_3)
        rotations = None
        if 'rot_0' in pcd_t.point and 'rot_1' in pcd_t.point and 'rot_2' in pcd_t.point and 'rot_3' in pcd_t.point:
            print("Found rotation quaternions")
            rot_0 = pcd_t.point['rot_0'].numpy()
            rot_1 = pcd_t.point['rot_1'].numpy()
            rot_2 = pcd_t.point['rot_2'].numpy()
            rot_3 = pcd_t.point['rot_3'].numpy()
            rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1)
            rotations = np.squeeze(rotations)
            if rotations.ndim == 3 and rotations.shape[1] == 1:
                rotations = rotations.squeeze(1)
            print(f"Rotation data shape: {rotations.shape}, range: [{rotations.min():.4f}, {rotations.max():.4f}]")
        else:
            print("No rotation quaternions found, using identity rotations")
            rotations = np.zeros((len(positions), 4))
            rotations[:, 0] = 1.0  # w=1, x=y=z=0 for identity quaternion
        
        # Extract opacity values
        opacity = None
        if 'opacity' in pcd_t.point:
            print("Found opacity values")
            opacity = pcd_t.point['opacity'].numpy()
            opacity = np.squeeze(opacity)
            if opacity.ndim == 2 and opacity.shape[1] == 1:
                opacity = opacity.squeeze(1)
            print(f"Opacity data shape: {opacity.shape}, range: [{opacity.min():.4f}, {opacity.max():.4f}]")
        else:
            print("No opacity values found, using full opacity")
            opacity = np.ones(len(positions))
        
        self.num_points = len(positions)
        
        # Store all Gaussian splatting data for CPU/GPU transformations
        self.original_positions = positions.astype(np.float32)
        self.colors = colors.astype(np.float32)
        self.scales = scales.astype(np.float32)
        self.rotations = rotations.astype(np.float32)
        self.opacities = opacity.astype(np.float32)
        
        # Debug summary of loaded Gaussian splatting data
        print("=== GAUSSIAN SPLATTING DATA SUMMARY ===")
        print(f"Positions: {self.original_positions.shape}")
        print(f"Colors: {self.colors.shape}")
        print(f"Scales: {self.scales.shape}")
        print(f"Rotations: {self.rotations.shape}")
        print(f"Opacities: {self.opacities.shape}")
        print("=== END DATA SUMMARY ===")
        
        # Create initial interleaved vertex data (will be updated with transformed data)
        vertex_data = np.zeros((len(positions), 6), dtype=np.float32)
        vertex_data[:, :3] = positions
        vertex_data[:, 3:6] = colors
        vertex_data = vertex_data.flatten()
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)  # Dynamic for updates
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def render(self, mv_matrix, p_matrix):
        """Render the points with given MV and P matrices using CuPy GPU transformation"""
        if not self.shader_program or not self.vao:
            return
        
        # Transform points using custom CUDA kernel (two-stage transformation)
        # Add homogeneous coordinate (w=1) to all points
        homogeneous_positions = np.ones((self.num_points, 4), dtype=np.float32)
        homogeneous_positions[:, :3] = self.original_positions
        
        # Transfer data to GPU
        gpu_homogeneous = cp.asarray(homogeneous_positions)
        gpu_mv = cp.asarray(mv_matrix.astype(np.float32))
        gpu_p = cp.asarray(p_matrix.astype(np.float32))
        
        # First transformation: Apply Model-View matrix (keep homogeneous coordinates)
        gpu_view_space = cp.zeros((self.num_points, 4), dtype=cp.float32)  # Keep homogeneous for intermediate
        
        # Launch first transform kernel (MV transformation, no perspective division)
        block_size = 256
        grid_size = (self.num_points + block_size - 1) // block_size
        self.transform_kernel((grid_size,), (block_size,), 
                            (gpu_mv, gpu_homogeneous, gpu_view_space, self.num_points))
        
        # Sort points by depth (z-coordinate in view space, farthest to closest for proper blending)
        view_z = gpu_view_space[:, 2]  # Extract z-coordinates in view space
        sorted_indices = cp.argsort(-view_z)  # Sort by negative z (farthest to closest)
        
        # Apply sorting to view space coordinates
        gpu_view_space_sorted = gpu_view_space[sorted_indices]
        
        # Apply sorting to all Gaussian attributes on GPU
        gpu_colors = cp.asarray(self.colors)
        gpu_colors_sorted = gpu_colors[sorted_indices]
        
        gpu_scales = cp.asarray(self.scales)
        gpu_scales_sorted = gpu_scales[sorted_indices]
        
        gpu_rotations = cp.asarray(self.rotations)
        gpu_rotations_sorted = gpu_rotations[sorted_indices]
        
        gpu_opacities = cp.asarray(self.opacities)
        gpu_opacities_sorted = gpu_opacities[sorted_indices]
        
        # Compute 2D covariance matrices and quad parameters
        gpu_cov2d = cp.zeros((self.num_points, 3), dtype=cp.float32)  # Upper triangle of 2x2 matrix
        gpu_quad_params = cp.zeros((self.num_points, 4), dtype=cp.float32)  # center_x, center_y, radius_x, radius_y
        gpu_visibility_mask = cp.zeros(self.num_points, dtype=cp.int32)
        
        self.covariance_kernel((grid_size,), (block_size,), 
                             (gpu_view_space_sorted, gpu_scales_sorted, gpu_rotations_sorted,
                              gpu_mv, gpu_p, gpu_cov2d, gpu_quad_params, gpu_visibility_mask, self.num_points))
        
        # Second transformation: Apply Projection matrix with perspective division (for point centers)
        gpu_transformed_positions = cp.zeros((self.num_points, 3), dtype=cp.float32)
        self.transform_perspective_kernel((grid_size,), (block_size,), 
                                        (gpu_p, gpu_view_space_sorted, gpu_transformed_positions, self.num_points))
        
        # Transfer back to CPU for rendering
        transformed_positions = cp.asnumpy(gpu_transformed_positions)
        colors_sorted = cp.asnumpy(gpu_colors_sorted)
        cov2d_data = cp.asnumpy(gpu_cov2d)
        quad_params = cp.asnumpy(gpu_quad_params)
        visibility_mask = cp.asnumpy(gpu_visibility_mask)
        
        # Debug output for first frame
        if not self._debug_printed:
            print("=== GAUSSIAN SPLATTING PIPELINE WITH 2D COVARIANCE ===")
            print("First 3 original points:", self.original_positions[:3])
            print("First 3 view-space points (unsorted):", cp.asnumpy(gpu_view_space[:3]))
            print("First 3 view-space z-values (sorted):", cp.asnumpy(view_z[sorted_indices[:3]]))
            print("First 3 transformed points (sorted):", transformed_positions[:3])
            print("First 3 2D covariance matrices:", cov2d_data[:3])
            print("First 3 quad parameters (center_x, center_y, radius_x, radius_y):", quad_params[:3])
            print("Visible Gaussians:", visibility_mask.sum(), "/", len(visibility_mask))
            print("Sorting indices (first 10):", cp.asnumpy(sorted_indices[:10]))
            self._debug_printed = True
        
        # Create interleaved vertex data with transformed positions and sorted colors
        vertex_data = np.zeros((self.num_points, 6), dtype=np.float32)
        vertex_data[:, :3] = transformed_positions
        vertex_data[:, 3:6] = colors_sorted
        vertex_data = vertex_data.flatten()
        
        # Update VBO with transformed data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        
        # Render using shader (no MVP uniform needed)
        glUseProgram(self.shader_program)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.shader_program:
            glDeleteProgram(self.shader_program)