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
        
        # Gaussian quad shaders
        self.gaussian_vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;      // Vertex position in NDC
        layout (location = 1) in vec3 aColor;    // Vertex color
        layout (location = 2) in vec2 aUV;       // UV coordinates
        layout (location = 3) in vec4 aQuadData; // Per-quad data (opacity, inv_cov_00, inv_cov_01, inv_cov_11)
        layout (location = 4) in vec2 aQuadRadii; // Per-quad radii (radius_x, radius_y)
        
        out vec3 fragColor;
        out vec2 fragUV;
        out vec4 quadData;
        out vec2 quadRadii;
        
        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            fragColor = aColor;
            fragUV = aUV;
            quadData = aQuadData;
            quadRadii = aQuadRadii;
        }
        """
        
        self.gaussian_fragment_shader_source = """
        #version 330 core
        in vec3 fragColor;
        in vec2 fragUV;
        in vec4 quadData;  // opacity, inv_cov_00, inv_cov_01, inv_cov_11
        in vec2 quadRadii; // radius_x, radius_y (NDC extents)
        
        out vec4 FragColor;
        
        void main()
        {
            float opacity = quadData.x;
            float inv_cov_00 = quadData.y;
            float inv_cov_01 = quadData.z;
            float inv_cov_11 = quadData.w;
            
            // DEBUG: Visualize radii data to see if it's being passed correctly
            if (fragUV.x < 0.33) {
                // Left third: Show radii as colors (scaled for visibility)
                float radius_vis_x = quadRadii.x * 100.0; // Scale up for visibility
                float radius_vis_y = quadRadii.y * 100.0;
                FragColor = vec4(radius_vis_x, radius_vis_y, 0.0, 1.0);
                return;
            } else if (fragUV.x < 0.66) {
                // Middle third: Show old behavior (fixed UV mapping)
                vec2 d_old = (fragUV - 0.5) * 6.0;
                float fixed_scale = 0.001;
                float exponent_old = -0.5 * (d_old.x * d_old.x * inv_cov_00 * fixed_scale + 
                                            2.0 * d_old.x * d_old.y * inv_cov_01 * fixed_scale + 
                                            d_old.y * d_old.y * inv_cov_11 * fixed_scale);
                exponent_old = max(exponent_old, -10.0);
                float alpha_old = opacity * exp(exponent_old);
                alpha_old = clamp(alpha_old, 0.0, 1.0);
                FragColor = vec4(fragColor, alpha_old);
                return;
            } else {
                // Right third: New mathematically correct approach
                vec2 d = (fragUV - 0.5) * 2.0 * quadRadii;
                
                float scaled_inv_cov_00 = inv_cov_00;
                float scaled_inv_cov_01 = inv_cov_01;
                float scaled_inv_cov_11 = inv_cov_11;
                
                float exponent = -0.5 * (d.x * d.x * scaled_inv_cov_00 + 
                                        2.0 * d.x * d.y * scaled_inv_cov_01 + 
                                        d.y * d.y * scaled_inv_cov_11);
                
                exponent = max(exponent, -10.0);
                float alpha = opacity * exp(exponent);
                alpha = clamp(alpha, 0.0, 1.0);
                FragColor = vec4(fragColor, alpha);
                return;
            }
        }
        """
        
        # OpenGL objects
        self.shader_program = None
        self.gaussian_shader_program = None  # Shader program for Gaussian quads
        self.vao = None
        self.vbo = None
        self.quad_vao = None        # VAO for quad rendering
        self.quad_vbo = None        # VBO for quad vertices
        self.quad_uv_vbo = None     # VBO for UV coordinates
        self.quad_data_vbo = None   # VBO for per-quad data
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
        self.quad_generation_kernel = None
        if CUPY_AVAILABLE:
            self._load_transform_kernel()
        else:
            raise RuntimeError("CuPy is required for the custom transform kernel")
        
        # Initialize if PLY file is provided
        if ply_path and os.path.exists(ply_path):
            self._setup_shaders()
            self._setup_gaussian_shaders()
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
    
    def _setup_gaussian_shaders(self):
        """Create and compile Gaussian quad shaders"""
        print("Setting up Gaussian quad shaders...")
        
        # Compile shaders
        vertex_shader = self._compile_shader(self.gaussian_vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(self.gaussian_fragment_shader_source, GL_FRAGMENT_SHADER)
        
        # Create shader program
        self.gaussian_shader_program = glCreateProgram()
        glAttachShader(self.gaussian_shader_program, vertex_shader)
        glAttachShader(self.gaussian_shader_program, fragment_shader)
        glLinkProgram(self.gaussian_shader_program)
        
        if not glGetProgramiv(self.gaussian_shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.gaussian_shader_program).decode()
            raise RuntimeError(f"Gaussian program linking failed: {error}")
        
        # Clean up shaders
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
    
    def _load_transform_kernel(self):
        """Load the custom CUDA transform kernels"""
        try:
            with open('transform_kernel.cu', 'r') as f:
                kernel_source = f.read()
            self.transform_kernel = cp.RawKernel(kernel_source, 'transform_points')
            self.transform_perspective_kernel = cp.RawKernel(kernel_source, 'transform_points_with_perspective')
            self.covariance_kernel = cp.RawKernel(kernel_source, 'compute_2d_covariance')
            self.quad_generation_kernel = cp.RawKernel(kernel_source, 'generate_quad_vertices')
            self.index_generation_kernel = cp.RawKernel(kernel_source, 'generate_quad_indices')
            self.compact_visible_kernel = cp.RawKernel(kernel_source, 'compact_visible_quads')
            print("Custom transform, covariance, quad generation, index generation, and compaction kernels loaded successfully")
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
            print(f"Scale data shape (log space): {scales.shape}, range: [{scales.min():.4f}, {scales.max():.4f}]")
            # Convert from log space to linear space
            scales = np.exp(scales)
            print(f"Scale data (linear space) range: [{scales.min():.6f}, {scales.max():.6f}]")
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
            # Note: PLY stores as (x,y,z,w) in rot_0-3, but we need (w,x,y,z) for the kernel
            rotations = np.stack([rot_3, rot_0, rot_1, rot_2], axis=-1)  # Reorder to (w,x,y,z)
            rotations = np.squeeze(rotations)
            if rotations.ndim == 3 and rotations.shape[1] == 1:
                rotations = rotations.squeeze(1)
            print(f"Rotation data shape (raw): {rotations.shape}, range: [{rotations.min():.4f}, {rotations.max():.4f}]")
            # Normalize quaternions to ensure they are unit quaternions
            norms = np.linalg.norm(rotations, axis=1, keepdims=True)
            rotations = rotations / norms
            print(f"Rotation data (normalized) range: [{rotations.min():.6f}, {rotations.max():.6f}]")
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
            print(f"Opacity data shape (logit space): {opacity.shape}, range: [{opacity.min():.4f}, {opacity.max():.4f}]")
            # Convert from logit space to probability using sigmoid
            opacity = 1.0 / (1.0 + np.exp(-opacity))
            print(f"Opacity data (probability space) range: [{opacity.min():.6f}, {opacity.max():.6f}]")
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
        # Only sort points that are in front of the camera (z < 0)
        sorted_indices = cp.argsort(view_z)  # Sort by z (closest to farthest, then we'll reverse for visible ones)
        
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
        gpu_quad_params = cp.zeros((self.num_points, 5), dtype=cp.float32)  # center_x, center_y, radius_x, radius_y, ndc_z
        gpu_visibility_mask = cp.zeros(self.num_points, dtype=cp.int32)
        
        # Get viewport dimensions for NDC conversion
        viewport = glGetIntegerv(GL_VIEWPORT)
        viewport_width = float(viewport[2])
        viewport_height = float(viewport[3])
        
        self.covariance_kernel((grid_size,), (block_size,), 
                             (gpu_view_space_sorted, gpu_scales_sorted, gpu_rotations_sorted,
                              gpu_mv, gpu_p, gpu_cov2d, gpu_quad_params, gpu_visibility_mask,
                              viewport_width, viewport_height, self.num_points))
        
        # Generate quad vertices for visible Gaussians
        max_quads = self.num_points  # Maximum possible number of quads
        gpu_quad_vertices = cp.zeros((max_quads * 4, 6), dtype=cp.float32)  # 4 vertices per quad, 6 floats per vertex
        gpu_quad_uvs = cp.zeros((max_quads * 4, 2), dtype=cp.float32)       # 4 vertices per quad, 2 UV per vertex
        gpu_quad_data = cp.zeros((max_quads, 6), dtype=cp.float32)          # Per-quad data (opacity + inverse covariance + radii)
        gpu_visible_count = cp.zeros(1, dtype=cp.int32)                    # Counter for visible quads
        
        self.quad_generation_kernel((grid_size,), (block_size,),
                                  (gpu_quad_params, gpu_cov2d, gpu_visibility_mask,
                                   gpu_colors_sorted, gpu_opacities_sorted,
                                   gpu_quad_vertices, gpu_quad_uvs, gpu_quad_data,
                                   gpu_visible_count, self.num_points))
        
        # Get the actual number of visible quads
        visible_count = int(cp.asnumpy(gpu_visible_count)[0])
        
        # Second transformation: Apply Projection matrix with perspective division (for point centers - for debugging)
        gpu_transformed_positions = cp.zeros((self.num_points, 3), dtype=cp.float32)
        self.transform_perspective_kernel((grid_size,), (block_size,), 
                                        (gpu_p, gpu_view_space_sorted, gpu_transformed_positions, self.num_points))
        
        # Use CUDA prefix sum for efficient visible quad compaction
        # Step 1: Compute prefix sum of visibility mask
        gpu_prefix_sum = cp.cumsum(gpu_visibility_mask) - gpu_visibility_mask
        visible_count = int(cp.sum(gpu_visibility_mask))
        
        if visible_count > 0:
            # Step 2: Allocate output buffers
            gpu_quad_vertices_compacted = cp.zeros((visible_count * 4, 6), dtype=cp.float32)
            gpu_quad_uvs_compacted = cp.zeros((visible_count * 4, 2), dtype=cp.float32)
            gpu_quad_data_compacted = cp.zeros((visible_count, 6), dtype=cp.float32)
            
            # Step 3: Launch compaction kernel
            self.compact_visible_kernel((grid_size,), (block_size,),
                                      (gpu_quad_vertices, gpu_quad_uvs, gpu_quad_data,
                                       gpu_visibility_mask, gpu_prefix_sum,
                                       gpu_quad_vertices_compacted, gpu_quad_uvs_compacted, gpu_quad_data_compacted,
                                       self.num_points))
            
            # Step 4: Transfer compacted data to CPU
            quad_vertices = cp.asnumpy(gpu_quad_vertices_compacted)
            quad_uvs = cp.asnumpy(gpu_quad_uvs_compacted)
            quad_data = cp.asnumpy(gpu_quad_data_compacted)
        else:
            quad_vertices = np.array([], dtype=np.float32)
            quad_uvs = np.array([], dtype=np.float32)
            quad_data = np.array([], dtype=np.float32)
        
        # Transfer debug data
        transformed_positions = cp.asnumpy(gpu_transformed_positions)
        colors_sorted = cp.asnumpy(gpu_colors_sorted)
        cov2d_data = cp.asnumpy(gpu_cov2d)
        quad_params = cp.asnumpy(gpu_quad_params)
        visibility_mask = cp.asnumpy(gpu_visibility_mask)
        
        # Debug output for first frame (simplified)
        if not self._debug_printed:
            print("=== QUAD RENDERING ACTIVE ===")
            print(f"Visible Gaussians: {visibility_mask.sum()} / {len(visibility_mask)}")
            print(f"Generated quads: {visible_count}")
            self._debug_printed = True
        
        # Set up quad VAO and VBOs if not already done
        if self.quad_vao is None:
            self._setup_quad_rendering()
        
        # Render quads if we have any visible ones
        if visible_count > 0:
            self._render_quads(quad_vertices, quad_uvs, quad_data, visible_count)
        
        # Render quads as points (new method that doesn't require index generation)
        # render_quads_as_points = False
        # if render_quads_as_points:
        #     opacities_sorted = cp.asnumpy(gpu_opacities_sorted)
        #     self._render_quads_as_points(quad_params, colors_sorted, opacities_sorted, visibility_mask)
        # else:
        #     # Original point rendering for comparison/debugging
        #     # Create interleaved vertex data with transformed positions and sorted colors
        #     vertex_data = np.zeros((self.num_points, 6), dtype=np.float32)
        #     vertex_data[:, :3] = transformed_positions
        #     vertex_data[:, 3:6] = colors_sorted
        #     vertex_data = vertex_data.flatten()
        
        #     # Update VBO with transformed data
        #     glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        #     glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        
        #     # Render using shader (no MVP uniform needed)
        #     glUseProgram(self.shader_program)
        
        #     glBindVertexArray(self.vao)
        #     glDrawArrays(GL_POINTS, 0, self.num_points)
        #     glBindVertexArray(0)
        
        #     glUseProgram(0)
    
    def _setup_quad_rendering(self):
        """Set up OpenGL objects for quad rendering"""
        # Create quad VAO and VBOs
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)           # Vertex positions and colors
        self.quad_uv_vbo = glGenBuffers(1)        # UV coordinates
        self.quad_data_vbo = glGenBuffers(1)      # Per-quad data part 1 (opacity, inverse covariance)
        self.quad_radii_vbo = glGenBuffers(1)     # Per-quad data part 2 (radii)
        
        glBindVertexArray(self.quad_vao)
        
        # Set up vertex position and color buffer (6 floats per vertex: x,y,z,r,g,b)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)           # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))  # Color
        glEnableVertexAttribArray(1)
        
        # Set up UV coordinates buffer (2 floats per vertex: u,v)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_uv_vbo)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * 4, None)           # UV coordinates
        glEnableVertexAttribArray(2)
        
        # Set up per-quad data buffer part 1 (4 floats per quad: opacity, inv_cov_00, inv_cov_01, inv_cov_11)
        # This data needs to be duplicated for each vertex of the quad
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_data_vbo)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * 4, None)           # Quad data part 1
        glEnableVertexAttribArray(3)
        
        # Set up per-quad data buffer part 2 (2 floats per quad: radius_x, radius_y)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_radii_vbo)
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 2 * 4, None)           # Quad radii
        glEnableVertexAttribArray(4)
        
        glBindVertexArray(0)
        print("Quad rendering setup complete")
    
    def _render_quads(self, quad_vertices, quad_uvs, quad_data, visible_count):
        """Render the generated quads"""
        # Update quad vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_DYNAMIC_DRAW)
        
        # Update UV buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_uv_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_uvs.nbytes, quad_uvs, GL_DYNAMIC_DRAW)
        
        # Split 6-component quad data into two parts
        quad_data_part1 = quad_data[:, :4]  # opacity, inv_cov (3 components)
        quad_radii_data = quad_data[:, 4:]   # radii (2 components)
        
        # Update per-quad data buffer part 1 - duplicate data for each vertex
        quad_data_part1_per_vertex = np.repeat(quad_data_part1, 4, axis=0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_data_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_data_part1_per_vertex.nbytes, quad_data_part1_per_vertex, GL_DYNAMIC_DRAW)
        
        # Update per-quad radii buffer - duplicate data for each vertex
        quad_radii_per_vertex = np.repeat(quad_radii_data, 4, axis=0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_radii_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_radii_per_vertex.nbytes, quad_radii_per_vertex, GL_DYNAMIC_DRAW)
        
        # Enable alpha blending for proper Gaussian splatting
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable depth testing but disable depth writing for alpha blending
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)  # Draw if fragment is closer to camera
        glDepthMask(GL_FALSE)  # Disable writing to depth buffer for transparent objects
        
        # Render quads using Gaussian shader
        glUseProgram(self.gaussian_shader_program)
        glBindVertexArray(self.quad_vao)
        
        # Draw as triangles (2 triangles per quad = 6 vertices per quad)
        # But we have 4 vertices per quad, so we need to use an index buffer or draw as triangle strip
        # For now, let's draw each quad as two triangles using GL_TRIANGLES with repeated vertices
        # We'll need to generate proper triangle indices
        
        # Generate indices on GPU using CUDA kernel
        num_indices = visible_count * 6  # 6 indices per quad (2 triangles)
        gpu_indices = cp.zeros(num_indices, dtype=cp.uint32)
        
        # Launch index generation kernel
        block_size = 256
        grid_size = (visible_count + block_size - 1) // block_size
        self.index_generation_kernel((grid_size,), (block_size,),
                                   (gpu_indices, visible_count))
        
        # Transfer indices to CPU
        indices = cp.asnumpy(gpu_indices)
        
        # Create and bind index buffer
        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Draw all quads with a single indexed draw call
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        # Clean up index buffer
        glDeleteBuffers(1, [ibo])
        
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)  # Clean up blending state
        glDepthMask(GL_TRUE)  # Restore depth writing
    
    def _render_quads_as_points(self, quad_params, colors_sorted, opacities_sorted, visibility_mask):
        """Render quad centers as points without index generation"""
        # Extract visible Gaussians only
        visible_indices = np.where(visibility_mask > 0)[0]
        if len(visible_indices) == 0:
            return
        
        # Extract quad centers and colors for visible Gaussians
        visible_centers = quad_params[visible_indices, :2]  # center_x, center_y from quad_params
        visible_z = quad_params[visible_indices, 4]  # ndc_z from quad_params
        visible_colors = colors_sorted[visible_indices]
        visible_opacities = opacities_sorted[visible_indices]
        
        # Create vertex data for points (x, y, z, r, g, b)
        point_count = len(visible_indices)
        vertex_data = np.zeros((point_count, 6), dtype=np.float32)
        vertex_data[:, 0] = visible_centers[:, 0]  # x in NDC
        vertex_data[:, 1] = visible_centers[:, 1]  # y in NDC
        vertex_data[:, 2] = visible_z  # z in NDC (with proper depth)
        vertex_data[:, 3:6] = visible_colors
        vertex_data = vertex_data.flatten()
        
        # Update VBO with point data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)
        
        # Enable alpha blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Render using shader
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        
        # Draw as points
        glDrawArrays(GL_POINTS, 0, point_count)
        
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
        if self.quad_vbo:
            glDeleteBuffers(1, [self.quad_vbo])
        if self.quad_uv_vbo:
            glDeleteBuffers(1, [self.quad_uv_vbo])
        if self.quad_data_vbo:
            glDeleteBuffers(1, [self.quad_data_vbo])
        if self.quad_radii_vbo:
            glDeleteBuffers(1, [self.quad_radii_vbo])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
        if self.gaussian_shader_program:
            glDeleteProgram(self.gaussian_shader_program)