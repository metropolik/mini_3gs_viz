from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians
import math
import os
import open3d as o3d
import ctypes
import transform
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
        
        # Fragment shader for Gaussian evaluation
        self.gaussian_fragment_shader_source = """
        #version 330 core
        in vec3 fragColor;
        in vec2 fragUV;
        in vec4 quadData;  // opacity, inv_cov_00, inv_cov_01, inv_cov_11
        in vec2 quadRadii; // radius_x, radius_y (NDC extents)
        in vec2 gaussianCenter; // Gaussian center in NDC space
        
        out vec4 FragColor;
        
        uniform vec2 viewport;  // Viewport dimensions (width, height)
        uniform int render_mod;  // Render mode: 0=gaussian, -2=billboard, -3=flat ball, -4=gaussian ball
        
        void main()
        {
            // Handle billboard mode first (like working method)
            if (render_mod == -2) {  // Billboard mode
                FragColor = vec4(fragColor, 1.0);
                return;
            }
            
            float opacity = quadData.x;
            float inv_cov_00 = quadData.y;
            float inv_cov_01 = quadData.z;
            float inv_cov_11 = quadData.w;
            
            // Convert gl_FragCoord from pixel coordinates to NDC
            vec2 fragNDC = (gl_FragCoord.xy / viewport) * 2.0 - 1.0;
            
            // Calculate distance from fragment to Gaussian center in NDC space
            vec2 d = fragNDC - gaussianCenter;
            
            // Evaluate Gaussian function
            float exponent = -0.5 * (d.x * d.x * inv_cov_00 + 
                                    2.0 * d.x * d.y * inv_cov_01 + 
                                    d.y * d.y * inv_cov_11);
            
            exponent = max(exponent, -10.0);
            float alpha = opacity * exp(exponent);
            alpha = clamp(alpha, 0.0, 1.0);
            
            // Handle rendering modes
            if (render_mod == -3) {  // Flat Ball
                alpha = alpha > 0.22 ? 1.0 : 0.0;
            } else if (render_mod == -4) {  // Gaussian Ball
                alpha = alpha > 0.22 ? 1.0 : 0.0;
                FragColor = vec4(fragColor * exp(exponent), alpha);
                return;
            }
            
            FragColor = vec4(fragColor, alpha);
        }
        """
        
        # Instanced rendering shaders
        self.instanced_vertex_shader_source = """
        #version 330 core
        // Per-vertex attributes (base quad)
        layout (location = 0) in vec2 aQuadVertex;  // Base quad vertex position (-1 to 1)
        
        // Per-instance attributes
        layout (location = 1) in vec3 aInstanceCenter;    // center_x, center_y, ndc_z
        layout (location = 2) in vec3 aInstanceColor;     // r, g, b
        layout (location = 3) in vec4 aInstanceCovariance; // opacity, inv_cov_00, inv_cov_01, inv_cov_11
        
        out vec3 fragColor;
        out vec2 gaussianCenter;
        out vec4 quadData;
        
        uniform vec2 viewport;  // Viewport dimensions for NDC conversion
        uniform float fovy;     // Field of view Y in radians
        uniform float aspect;   // Aspect ratio (width/height)
        
        void main()
        {
            // Extract inverse covariance matrix
            float inv_cov_00 = aInstanceCovariance.y;
            float inv_cov_01 = aInstanceCovariance.z;
            float inv_cov_11 = aInstanceCovariance.w;
            
            // Convert inverse covariance back to 2D covariance (like working method)
            float det_inv = inv_cov_00 * inv_cov_11 - inv_cov_01 * inv_cov_01;
            if (abs(det_inv) < 1e-6) det_inv = 1e-6;
            
            float cov2d_00 = inv_cov_11 / det_inv;
            float cov2d_11 = inv_cov_00 / det_inv;
            
            // Use the working method approach: 3 * sqrt(cov2d diagonal elements)
            vec2 quadwh_scr = vec2(3.0 * sqrt(max(cov2d_00, 1e-6)), 3.0 * sqrt(max(cov2d_11, 1e-6)));
            
            // Convert screen space dimensions to NDC space like working method
            // Working method: quadwh_ndc = quadwh_scr / wh * 2
            // where wh = 2 * hfovxy_focal.xy * hfovxy_focal.z
            
            // Derive equivalent values:
            // tan(fovy/2) = half FOV tangent for Y
            // tan(fovx/2) = tan(fovy/2) * aspect for X  
            float tan_half_fovy = tan(fovy * 0.5);
            float tan_half_fovx = tan_half_fovy * aspect;
            
            // Focal lengths (from projection matrix diagonal elements)
            float focal_y = 1.0 / tan_half_fovy;
            float focal_x = focal_y / aspect;
            
            // Working method's wh = 2 * [tan_half_fovx, tan_half_fovy] * focal
            // Simplified: wh = 2 * [tan_half_fovx * focal_x, tan_half_fovy * focal_y] = [2, 2]
            vec2 wh = 2.0 * vec2(tan_half_fovx * focal_x, tan_half_fovy * focal_y);
            
            vec2 quadwh_ndc = quadwh_scr / wh * 2.0;
            
            // Apply quad offset in NDC space without rotation
            vec3 position;
            position.xy = aInstanceCenter.xy + aQuadVertex * quadwh_ndc;
            position.z = aInstanceCenter.z;
            
            gl_Position = vec4(position, 1.0);
            
            fragColor = aInstanceColor;
            gaussianCenter = aInstanceCenter.xy;
            quadData = aInstanceCovariance;
        }
        """
        
        # OpenGL objects
        self.shader_program = None
        self.instanced_shader_program = None  # Shader program for instanced rendering
        self.vao = None
        self.vbo = None
        self.instance_vbo = None    # VBO for instance data
        self.base_quad_vbo = None   # VBO for base quad geometry
        self.instanced_vao = None   # VAO for instanced rendering
        self.num_points = 0
        
        # CPU transformation objects
        self.original_positions = None  # Original positions (CPU)
        self.colors = None              # Colors (CPU)
        self.scales = None              # Scale vectors (CPU)
        self.rotations = None           # Rotation quaternions (CPU)
        self.opacity = None             # Opacity values (CPU)
        self.use_cuda = False  # Disable CUDA for now
        
        # Render mode control
        self.render_mode = 0  # 0=gaussian, -3=flat ball, -4=gaussian ball
        
        # No longer need CUDA kernels - using Python implementations
        self.transform_kernel = None
        self.transform_perspective_kernel = None
        self.covariance_kernel = None
        self.quad_generation_kernel = None
        
        # Initialize if PLY file is provided, otherwise use hardcoded Gaussians
        if ply_path and os.path.exists(ply_path):
            self._setup_shaders()
            self._setup_instanced_shaders()
            self._load_ply()
            # Enable point size control
            glEnable(GL_PROGRAM_POINT_SIZE)
        elif ply_path:
            print(f"PLY file '{ply_path}' not found, using hardcoded Gaussians for idle visualization")
            self._setup_shaders()
            self._setup_instanced_shaders()
            self._create_naive_gaussians()
            # Enable point size control
            glEnable(GL_PROGRAM_POINT_SIZE)
        else:
            print("No PLY file provided, using hardcoded Gaussians for idle visualization")
            self._setup_shaders()
            self._setup_instanced_shaders()
            self._create_naive_gaussians()
            # Enable point size control
            glEnable(GL_PROGRAM_POINT_SIZE)
    
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
    
    
    def _setup_instanced_shaders(self):
        """Create and compile instanced rendering shaders"""
        print("Setting up instanced rendering shaders...")
        
        # Compile shaders
        vertex_shader = self._compile_shader(self.instanced_vertex_shader_source, GL_VERTEX_SHADER)
        # Use the same fragment shader as the regular Gaussian shader
        fragment_shader = self._compile_shader(self.gaussian_fragment_shader_source, GL_FRAGMENT_SHADER)
        
        # Create shader program
        self.instanced_shader_program = glCreateProgram()
        glAttachShader(self.instanced_shader_program, vertex_shader)
        glAttachShader(self.instanced_shader_program, fragment_shader)
        glLinkProgram(self.instanced_shader_program)
        
        if not glGetProgramiv(self.instanced_shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.instanced_shader_program).decode()
            raise RuntimeError(f"Instanced program linking failed: {error}")
        
        # Clean up shaders
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        # Get uniform locations
        self.instanced_viewport_uniform = glGetUniformLocation(self.instanced_shader_program, "viewport")
        self.instanced_render_mod_uniform = glGetUniformLocation(self.instanced_shader_program, "render_mod")
        self.instanced_fovy_uniform = glGetUniformLocation(self.instanced_shader_program, "fovy")
        self.instanced_aspect_uniform = glGetUniformLocation(self.instanced_shader_program, "aspect")
    
    
    
    def _load_ply(self):
        """Load PLY file and extract positions and colors"""
        print(f"Loading Gaussian splat model from {self.ply_path}")
        
        # Load PLY file using Open3D tensor API
        pcd_t = o3d.t.io.read_point_cloud(self.ply_path)
        
        # Get available field names
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
                    # Check for spherical harmonics fields
                    if field_name.startswith('f_dc_') or field_name.startswith('f_rest_'):
                        sh_fields.append(field_name)
                elif field_name in pcd_t.point:
                    field_data = pcd_t.point[field_name]
                    available_fields.append(field_name)
                    # Check for spherical harmonics fields
                    if field_name.startswith('f_dc_') or field_name.startswith('f_rest_'):
                        sh_fields.append(field_name)
            except:
                continue
        
        
        
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
            # Convert from log space to linear space
            scales = np.exp(scales)
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
            # Normalize quaternions to ensure they are unit quaternions
            norms = np.linalg.norm(rotations, axis=1, keepdims=True)
            rotations = rotations / norms
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
            # Convert from logit space to probability using sigmoid
            opacity = 1.0 / (1.0 + np.exp(-opacity))
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
    
    def _create_naive_gaussians(self):
        """Create hardcoded Gaussians matching the working method's naive_gaussian() function"""
        print("Creating naive hardcoded Gaussians for idle visualization")
        
        # Exactly matching working_method/util_gau.py naive_gaussian() function
        self.num_points = 4
        
        # Positions (xyz): origin, and three axis-aligned positions
        self.original_positions = np.array([
            [0.0, 0.0, 0.0],  # Origin
            [1.0, 0.0, 0.0],  # X-axis
            [0.0, 1.0, 0.0],  # Y-axis  
            [0.0, 0.0, 1.0]   # Z-axis
        ], dtype=np.float32)
        
        # Rotation quaternions (w, x, y, z): all identity rotations
        self.rotations = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Scale vectors: small default with axis-aligned stretching
        self.scales = np.array([
            [0.03, 0.03, 0.03],  # Small sphere
            [0.2, 0.03, 0.03],   # Stretched along X
            [0.03, 0.2, 0.03],   # Stretched along Y
            [0.03, 0.03, 0.2]    # Stretched along Z
        ], dtype=np.float32)
        
        # Colors (spherical harmonics DC term, converted to RGB)
        # Original working method colors: [1,0,1], [1,0,0], [0,1,0], [0,0,1]
        gau_c = np.array([
            [1.0, 0.0, 1.0],  # Magenta
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0]   # Blue
        ], dtype=np.float32)
        # Convert from SH DC term: (color - 0.5) / 0.28209 -> color
        # So we reverse: color * 0.28209 + 0.5
        self.colors = gau_c * 0.28209479177387814 + 0.5
        
        # Opacity: all fully opaque
        self.opacities = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        # Create initial interleaved vertex data for OpenGL
        vertex_data = np.zeros((self.num_points, 6), dtype=np.float32)
        vertex_data[:, :3] = self.original_positions
        vertex_data[:, 3:6] = self.colors
        vertex_data = vertex_data.flatten()
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)
        
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
        
        # Transform points using Python implementation (two-stage transformation)
        # Add homogeneous coordinate (w=1) to all points
        homogeneous_positions = np.ones((self.num_points, 4), dtype=np.float32)
        homogeneous_positions[:, :3] = self.original_positions
        
        # Transfer data to GPU
        gpu_homogeneous = cp.asarray(homogeneous_positions)
        gpu_mv = cp.asarray(mv_matrix.astype(np.float32))
        gpu_p = cp.asarray(p_matrix.astype(np.float32))
        
        # Use Python implementation instead of CUDA kernel for first transformation
        # First transformation: Apply Model-View matrix (keep homogeneous coordinates)
        view_space = np.zeros((self.num_points, 4), dtype=np.float32)
        view_space_flat = view_space.reshape(-1)  # Use reshape instead of flatten for a view
        transform.transform_points(mv_matrix.flatten().astype(np.float32), 
                                 homogeneous_positions.flatten(), 
                                 view_space_flat, 
                                 self.num_points)
        
        # Convert back to GPU for the remaining operations
        gpu_view_space = cp.asarray(view_space)
        
        # Set up CUDA grid parameters for remaining kernels
        block_size = 256
        grid_size = (self.num_points + block_size - 1) // block_size
        
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
        
        # Use Python implementation instead of CUDA kernel
        cov2d = np.zeros((self.num_points, 3), dtype=np.float32)
        quad_params = np.zeros((self.num_points, 5), dtype=np.float32)
        visibility_mask = np.zeros(self.num_points, dtype=np.int32)
        
        view_space_sorted_np = cp.asnumpy(gpu_view_space_sorted)
        scales_sorted_np = cp.asnumpy(gpu_scales_sorted)
        rotations_sorted_np = cp.asnumpy(gpu_rotations_sorted)
        
        transform.compute_2d_covariance(view_space_sorted_np.reshape(-1),
                                      scales_sorted_np.reshape(-1),
                                      rotations_sorted_np.reshape(-1),
                                      mv_matrix.reshape(-1).astype(np.float32),
                                      p_matrix.reshape(-1).astype(np.float32),
                                      cov2d.reshape(-1),
                                      quad_params.reshape(-1),
                                      visibility_mask.reshape(-1),
                                      viewport_width, viewport_height, self.num_points)
        
        gpu_cov2d = cp.asarray(cov2d)
        gpu_quad_params = cp.asarray(quad_params)
        gpu_visibility_mask = cp.asarray(visibility_mask)
        
        # Skip quad generation when using instanced rendering
        visible_count = self.num_points  # All quads are visible
        
        # Generate instance data for future instanced rendering (Step 3)
        # Use Python implementation instead of CUDA kernel
        instance_data = np.zeros((self.num_points, 10), dtype=np.float32)
        colors_sorted_np = cp.asnumpy(gpu_colors_sorted)
        opacities_sorted_np = cp.asnumpy(gpu_opacities_sorted)
        
        transform.generate_instance_data(quad_params.reshape(-1),
                                       cov2d.reshape(-1),
                                       visibility_mask.reshape(-1),
                                       colors_sorted_np.reshape(-1),
                                       opacities_sorted_np.reshape(-1),
                                       instance_data.reshape(-1),
                                       self.num_points)
        
        gpu_instance_data = cp.asarray(instance_data)
        
        # Second transformation: Apply Projection matrix with perspective division (for point centers - for debugging)
        # Use Python implementation instead of CUDA kernel
        transformed_positions = np.zeros((self.num_points, 3), dtype=np.float32)
        transformed_positions_flat = transformed_positions.reshape(-1)
        view_space_sorted_np = cp.asnumpy(gpu_view_space_sorted)
        transform.transform_points_with_perspective(p_matrix.flatten().astype(np.float32),
                                                  view_space_sorted_np.flatten(),
                                                  transformed_positions_flat,
                                                  self.num_points)
        gpu_transformed_positions = cp.asarray(transformed_positions)
        
        # No need to transfer quad data when using instanced rendering
        
        
        
        # Create instance VBO if not already done (Step 3)
        if self.instance_vbo is None:
            self.instance_vbo = glGenBuffers(1)
        
        # Update instance buffer with all instance data (no visibility check for performance)
        instance_data = cp.asnumpy(gpu_instance_data).flatten()
        
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Set up instanced rendering if not done
        if self.instanced_vao is None:
            self._setup_instanced_rendering()
        
        # Render using instanced rendering
        self._render_instanced(self.num_points, viewport_width, viewport_height)
    
    
    def _setup_instanced_rendering(self):
        """Set up OpenGL objects for instanced rendering"""
        if self.instanced_vao is not None:
            return  # Already set up
            
        print("Setting up instanced rendering...")
        
        # Create base quad vertices (2D positions)
        base_quad = np.array([
            [-1.0, -1.0],  # Bottom-left
            [ 1.0, -1.0],  # Bottom-right
            [-1.0,  1.0],  # Top-left
            [ 1.0,  1.0],  # Top-right
        ], dtype=np.float32).flatten()
        
        # Create indices for two triangles
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        
        # Create VAO and VBOs
        self.instanced_vao = glGenVertexArrays(1)
        self.base_quad_vbo = glGenBuffers(1)
        self.base_quad_ibo = glGenBuffers(1)
        
        glBindVertexArray(self.instanced_vao)
        
        # Set up base quad vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.base_quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, base_quad.nbytes, base_quad, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Set up index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.base_quad_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Set up instance buffer attributes
        if self.instance_vbo is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
            
            # Instance data layout: 10 floats per instance
            # center_x, center_y, ndc_z, r, g, b, opacity, inv_cov_00, inv_cov_01, inv_cov_11
            stride = 10 * 4  # 10 floats * 4 bytes
            
            # Center position (location = 1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribDivisor(1, 1)  # One per instance
            
            # Color (location = 2)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(2)
            glVertexAttribDivisor(2, 1)  # One per instance
            
            # Covariance data (location = 3)
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
            glEnableVertexAttribArray(3)
            glVertexAttribDivisor(3, 1)  # One per instance
        
        glBindVertexArray(0)
        print("Instanced rendering setup complete")
    
    
    def _render_instanced(self, num_instances, viewport_width, viewport_height):
        """Render using instanced rendering"""
        # Enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable depth testing but disable depth writing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_FALSE)
        
        # Use instanced shader program
        glUseProgram(self.instanced_shader_program)
        
        # Set uniforms
        glUniform2f(self.instanced_viewport_uniform, viewport_width, viewport_height)
        glUniform1i(self.instanced_render_mod_uniform, self.render_mode)
        glUniform1f(self.instanced_fovy_uniform, math.radians(45.0))  # 45 degrees in radians
        glUniform1f(self.instanced_aspect_uniform, viewport_width / viewport_height)
        
        # Bind VAO and render
        glBindVertexArray(self.instanced_vao)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        glBindVertexArray(0)
        
        glUseProgram(0)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
    
    def set_render_mode(self, mode):
        """Set rendering mode: 0=gaussian, -2=billboard, -3=flat ball, -4=gaussian ball"""
        self.render_mode = mode
    
    def set_gaussian_mode(self):
        """Set to normal Gaussian splatting mode"""
        self.render_mode = 0
    
    def set_billboard_mode(self):
        """Set to billboard rendering mode"""
        self.render_mode = -2
    
    def set_flat_ball_mode(self):
        """Set to flat ball rendering mode"""
        self.render_mode = -3
    
    def set_gaussian_ball_mode(self):
        """Set to Gaussian ball rendering mode"""
        self.render_mode = -4
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.instance_vbo:
            glDeleteBuffers(1, [self.instance_vbo])
        if self.base_quad_vbo:
            glDeleteBuffers(1, [self.base_quad_vbo])
        if hasattr(self, 'base_quad_ibo') and self.base_quad_ibo:
            glDeleteBuffers(1, [self.base_quad_ibo])
        if self.instanced_vao:
            glDeleteVertexArrays(1, [self.instanced_vao])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
        if self.instanced_shader_program:
            glDeleteProgram(self.instanced_shader_program)