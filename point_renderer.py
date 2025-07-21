from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians
import math
import os
import open3d as o3d
import ctypes

class PointRenderer:
    def __init__(self, ply_path=None):
        """Initialize point renderer with optional PLY file
        
        Args:
            ply_path: Path to PLY file
        """
        self.ply_path = ply_path
        
        
        # Copy working method's fragment shader exactly
        self.gaussian_fragment_shader_source = """
        #version 430 core

        in vec3 color;
        in float alpha;
        in vec3 conic;
        in vec2 coordxy;  // local coordinate in quad, unit in pixel

        uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball

        out vec4 FragColor;

        void main()
        {
            if (render_mod == -2)
            {
                FragColor = vec4(color, 1.f);
                return;
            }

            float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
            //FragColor.rgba = vec4(vec3(abs(power) * 0.1), 1.0);  // Power values (should be smooth)
            if (power > 0.f)
                discard;
            float opacity = min(0.99f, alpha * exp(power));
            if (opacity < 1.f / 255.f)
                discard;
            //FragColor.rgba = vec4(vec3(opacity), 1.0);
            //FragColor.rgba = vec4(abs(conic.x) * 0.1, abs(conic.y) * 0.1, abs(conic.z) * 0.1, 1.0);  // Conic values
            //FragColor.rgba = vec4(abs(coordxy.x) / 50.0, abs(coordxy.y) / 50.0, 0.5, 1.0);  // Coord values
            //FragColor.rgba = vec4(abs(conic.x) * 5.0, abs(conic.y) * 5.0, abs(conic.z) * 5.0, 1.0);  // Conic values (brighter)
            //return;
            FragColor = vec4(color, 1.0);
            //FragColor = vec4(color, opacity);

            // handling special shading effect
            if (render_mod == -3)
                FragColor.a = FragColor.a > 0.22 ? 1 : 0;
            else if (render_mod == -4)
            {
                FragColor.a = FragColor.a > 0.22 ? 1 : 0;
                FragColor.rgb = FragColor.rgb * exp(power);
            }
        }
        """
        
        # Copy working method's vertex shader exactly
        self.instanced_vertex_shader_source = """
        #version 430 core

        #define SH_C0 0.28209479177387814f
        #define SH_C1 0.4886025119029199f
        #define SH_C2_0 1.0925484305920792f
        #define SH_C2_1 -1.0925484305920792f
        #define SH_C2_2 0.31539156525252005f
        #define SH_C2_3 -1.0925484305920792f
        #define SH_C2_4 0.5462742152960396f
        #define SH_C3_0 -0.5900435899266435f
        #define SH_C3_1 2.890611442640554f
        #define SH_C3_2 -0.4570457994644658f
        #define SH_C3_3 0.3731763325901154f
        #define SH_C3_4 -0.4570457994644658f
        #define SH_C3_5 1.445305721320277f
        #define SH_C3_6 -0.5900435899266435f

        layout(location = 0) in vec2 position;

        #define POS_IDX 0
        #define ROT_IDX 3
        #define SCALE_IDX 7
        #define OPACITY_IDX 10
        #define SH_IDX 11

        layout (std430, binding=0) buffer gaussian_data {
            float g_data[];
        };
        layout (std430, binding=1) buffer gaussian_order {
            int gi[];
        };

        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform vec3 hfovxy_focal;
        uniform vec3 cam_pos;
        uniform int sh_dim;
        uniform float scale_modifier;
        uniform int render_mod;

        out vec3 color;
        out float alpha;
        out vec3 conic;
        out vec2 coordxy;

        mat3 computeCov3D(vec3 scale, vec4 q)
        {
            mat3 S = mat3(0.f);
            S[0][0] = scale.x;
            S[1][1] = scale.y;
            S[2][2] = scale.z;
            float r = q.x;
            float x = q.y;
            float y = q.z;
            float z = q.w;

            mat3 R = mat3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
            );

            mat3 M = S * R;
            mat3 Sigma = transpose(M) * M;
            return Sigma;
        }

        vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
        {
            vec4 t = mean_view;
            float limx = 1.3f * tan_fovx;
            float limy = 1.3f * tan_fovy;
            float txtz = t.x / t.z;
            float tytz = t.y / t.z;
            t.x = min(limx, max(-limx, txtz)) * t.z;
            t.y = min(limy, max(-limy, tytz)) * t.z;

            mat3 J = mat3(
                focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
                0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
                0, 0, 0
            );
            mat3 W = transpose(mat3(viewmatrix));
            mat3 T = W * J;

            mat3 cov = transpose(T) * transpose(cov3D) * T;
            cov[0][0] += 0.3f;
            cov[1][1] += 0.3f;
            return vec3(cov[0][0], cov[0][1], cov[1][1]);
        }

        vec3 get_vec3(int offset)
        {
            return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
        }
        vec4 get_vec4(int offset)
        {
            return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
        }

        void main()
        {
            int boxid = gi[gl_InstanceID];
            int total_dim = 3 + 4 + 3 + 1 + sh_dim;
            int start = boxid * total_dim;
            vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
            vec4 g_pos_view = view_matrix * g_pos;
            vec4 g_pos_screen = projection_matrix * g_pos_view;
            g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
            g_pos_screen.w = 1.f;
            
            // early culling
            if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))))
            {
                gl_Position = vec4(-100, -100, -100, 1);
                return;
            }
            vec4 g_rot = get_vec4(start + ROT_IDX);
            vec3 g_scale = get_vec3(start + SCALE_IDX);
            float g_opacity = g_data[start + OPACITY_IDX];

            mat3 cov3d = computeCov3D(g_scale * scale_modifier, g_rot);
            vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
            vec3 cov2d = computeCov2D(g_pos_view, 
                                      hfovxy_focal.z, 
                                      hfovxy_focal.z, 
                                      hfovxy_focal.x, 
                                      hfovxy_focal.y, 
                                      cov3d, 
                                      view_matrix);

            // Invert covariance (EWA algorithm)
            float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
            if (det == 0.0f)
                gl_Position = vec4(0.f, 0.f, 0.f, 0.f);
            
            float det_inv = 1.f / det;
            conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
            
            vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));
            vec2 quadwh_ndc = quadwh_scr / wh * 2;
            g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
            coordxy = position * quadwh_scr;
            gl_Position = g_pos_screen;
            
            alpha = g_opacity;

            //if (render_mod == -1)
            //{
            //    float depth = -g_pos_view.z;
            //    depth = depth < 0.05 ? 1 : depth;
            //    depth = 1 / depth;
            //    color = vec3(depth, depth, depth);
            //    return;
            //}

            // Convert SH to color (exactly like working method)
            int sh_start = start + SH_IDX;
            vec3 dir = g_pos.xyz - cam_pos;
            dir = normalize(dir);
            //color = vec3(cov3d[0][0] * 100.0, cov3d[1][1] * 100.0, cov3d[2][2] * 100.0);
            //color = vec3(abs(cov3d[0][1]) * 10000.0, abs(cov3d[0][2]) * 10000.0, abs(cov3d[1][2]) * 10000.0);
            color = vec3(cov2d.x * 0.01, abs(cov2d.y) * 0.01, cov2d.z * 0.001);  // 2D Covariance (high contrast)
            //color = SH_C0 * get_vec3(sh_start);
            
            //if (sh_dim > 3 && render_mod >= 1)
            //{
            //    float x = dir.x;
            //    float y = dir.y;
            //    float z = dir.z;
            //    color = color - SH_C1 * y * get_vec3(sh_start + 1 * 3) + SH_C1 * z * get_vec3(sh_start + 2 * 3) - SH_C1 * x * get_vec3(sh_start + 3 * 3);

            //    if (sh_dim > 12 && render_mod >= 2)
            //    {
            //        float xx = x * x, yy = y * y, zz = z * z;
            //        float xy = x * y, yz = y * z, xz = x * z;
            //        color = color +
            //            SH_C2_0 * xy * get_vec3(sh_start + 4 * 3) +
            //            SH_C2_1 * yz * get_vec3(sh_start + 5 * 3) +
            //            SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_start + 6 * 3) +
            //            SH_C2_3 * xz * get_vec3(sh_start + 7 * 3) +
            //            SH_C2_4 * (xx - yy) * get_vec3(sh_start + 8 * 3);

            //        if (sh_dim > 27 && render_mod >= 3)
            //        {
            //            color = color +
            //                SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_start + 9 * 3) +
            //                SH_C3_1 * xy * z * get_vec3(sh_start + 10 * 3) +
            //                SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_start + 11 * 3) +
            //                SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_start + 12 * 3) +
            //                SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_start + 13 * 3) +
            //                SH_C3_5 * z * (xx - yy) * get_vec3(sh_start + 14 * 3) +
            //                SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_start + 15 * 3);
            //        }
            //    }
            //}
            //color += 0.5f;
            // color = vec3(conic.x * 5.0, abs(conic.y) * 5.0, conic.z * 5.0);  // Conic coefficients (match fragment) // darker than blender
            // color = vec3(abs(coordxy.x) / 50.0, abs(coordxy.y) / 50.0, 0.5);  // Local coordinates  // seems to match
            // color = vec3(quadwh_scr.x / 100.0, quadwh_scr.y / 100.0, 0.0);   // Quad screen size // seems to match
            //color = vec3(abs(cov2d.x) * 100.0, abs(cov2d.y) * 100.0, abs(cov2d.z) * 100.0);  // 2D Covariance
            //color = vec3(tan_fovx * 2.0, tan_fovy * 2.0, focal_z / 1000.0);  // Focal parameters
        }
        """
        
        # OpenGL objects - storage buffer rendering (like working method)
        self.instanced_shader_program = None  # Shader program for rendering
        self.gaussian_data_buffer = None      # Storage buffer for gaussian data
        self.gaussian_order_buffer = None     # Storage buffer for sorted indices
        self.base_quad_vbo = None             # VBO for base quad geometry
        self.instanced_vao = None             # VAO for rendering
        self.num_points = 0
        
        # Gaussian data storage (like working method)
        self.gaussian_data = None       # Flattened gaussian data [pos,rot,scale,opacity,sh] * num_gaussians
        self.sorted_indices = None      # Sorted indices for depth sorting
        self.original_positions = None  # Original positions (CPU)
        self.colors = None              # Colors (CPU) - for fallback
        self.scales = None              # Scale vectors (CPU)
        self.rotations = None           # Rotation quaternions (CPU)
        self.opacities = None           # Opacity values (CPU)
        self.sh_coeffs = None           # SH coefficients (CPU)
        self.use_cuda = False           # Disable CUDA for now
        
        # Render mode control
        self.render_mode = 0  # 0=gaussian, -3=flat ball, -4=gaussian ball
        
        # No longer need CUDA kernels - using Python implementations
        self.transform_kernel = None
        self.transform_perspective_kernel = None
        self.covariance_kernel = None
        self.quad_generation_kernel = None
        
        # SH rendering parameters
        self.sh_render_mode = 0  # 0=DC only, 1=DC+degree1, 2=DC+degree1+2, 3=DC+degree1+2+3
        
        # Initialize if PLY file is provided, otherwise use hardcoded Gaussians
        if ply_path and os.path.exists(ply_path):
            self._setup_instanced_shaders()
            self._load_ply()
        elif ply_path:
            print(f"PLY file '{ply_path}' not found, using hardcoded Gaussians for idle visualization")
            self._setup_instanced_shaders()
            self._create_naive_gaussians()
        else:
            print("No PLY file provided, using hardcoded Gaussians for idle visualization")
            self._setup_instanced_shaders()
            self._create_naive_gaussians()
    
    def _compile_shader(self, source, shader_type):
        """Compile a shader from source"""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation failed: {error}")
        
        return shader
    
    def _setup_instanced_shaders(self):
        """Create and compile instanced rendering shaders"""
        print("Setting up instanced rendering shaders...")
        
        # Check GPU vertex attribute limits
        max_vertex_attribs = glGetIntegerv(GL_MAX_VERTEX_ATTRIBS)
        print(f"GPU supports {max_vertex_attribs} vertex attributes (we need 17 for full SH)")
        
        if max_vertex_attribs >= 17:
            print("✅ GPU can handle full SH instanced rendering")
        else:
            print(f"⚠️  GPU may not support full SH instanced rendering (need 17, have {max_vertex_attribs})")
        
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
        
        # Get uniform locations (matching working method)
        self.instanced_view_matrix_uniform = glGetUniformLocation(self.instanced_shader_program, "view_matrix")
        self.instanced_projection_matrix_uniform = glGetUniformLocation(self.instanced_shader_program, "projection_matrix")
        self.instanced_hfovxy_focal_uniform = glGetUniformLocation(self.instanced_shader_program, "hfovxy_focal")
        self.instanced_cam_pos_uniform = glGetUniformLocation(self.instanced_shader_program, "cam_pos")
        self.instanced_scale_modifier_uniform = glGetUniformLocation(self.instanced_shader_program, "scale_modifier")
        self.instanced_render_mod_uniform = glGetUniformLocation(self.instanced_shader_program, "render_mod")
        self.instanced_sh_dim_uniform = glGetUniformLocation(self.instanced_shader_program, "sh_dim")
        
        print(f"Shader uniforms: view_matrix={self.instanced_view_matrix_uniform}, sh_dim={self.instanced_sh_dim_uniform}")
    
    def _create_storage_buffer(self, name, data, binding_point, buffer_id=None):
        """Create or update a storage buffer (SSBO)"""
        if buffer_id is None:
            buffer_id = glGenBuffers(1)
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, buffer_id)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        return buffer_id
    
    def _create_gaussian_data_buffer(self):
        """Create the flattened gaussian data like working method"""
        if self.sh_coeffs is None:
            print("Error: No SH coefficients available for storage buffer")
            return
            
        # Create flattened data: [pos(3) + rot(4) + scale(3) + opacity(1) + sh(sh_dim)] * num_gaussians
        total_dim = 3 + 4 + 3 + 1 + self.sh_dim
        self.gaussian_data = np.zeros((self.num_points, total_dim), dtype=np.float32)
        
        # Fill data exactly like working method
        self.gaussian_data[:, 0:3] = self.original_positions      # pos at 0
        self.gaussian_data[:, 3:7] = self.rotations               # rot at 3  
        self.gaussian_data[:, 7:10] = self.scales                 # scale at 7
        self.gaussian_data[:, 10] = self.opacities                # opacity at 10
        self.gaussian_data[:, 11:11+self.sh_dim] = self.sh_coeffs # sh at 11
        
        # Flatten for storage buffer
        gaussian_data_flat = self.gaussian_data.flatten()
        
        # Create storage buffer
        self.gaussian_data_buffer = self._create_storage_buffer(
            "gaussian_data", gaussian_data_flat, 0, self.gaussian_data_buffer)
        
        print(f"Created gaussian data buffer: {self.num_points} gaussians, {total_dim} floats each, {gaussian_data_flat.nbytes} bytes total")
        print(f"First gaussian data sample: pos={self.gaussian_data[0, 0:3]}, sh_start={self.gaussian_data[0, 11:14]}")
        print(f"Buffer layout verification: POS=0, ROT=3, SCALE=7, OPACITY=10, SH=11")
    
    def _create_sorted_indices_buffer(self):
        """Create initial sorted indices buffer"""
        # Initialize with identity order (will be sorted later)
        self.sorted_indices = np.arange(self.num_points, dtype=np.int32)
        
        # Create storage buffer for indices
        self.gaussian_order_buffer = self._create_storage_buffer(
            "gaussian_order", self.sorted_indices, 1, self.gaussian_order_buffer)
        
        print(f"Created gaussian order buffer: {self.num_points} indices, {self.sorted_indices.nbytes} bytes")
    
    
    
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
        
        # Also check for f_rest_ fields for higher order SH
        for i in range(45):  # Support up to degree 3 SH (45 f_rest_ fields)
            test_fields.append(f'f_rest_{i}')
        
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
        
        # Extract spherical harmonics coefficients (f_dc_* and f_rest_*)
        # Support up to degree 3 SH like the working method
        max_sh_degree = 3
        sh_coeffs = None
        
        if 'f_dc_0' in pcd_t.point:
            print("Found spherical harmonics data")
            
            # Load DC terms (degree 0)
            dc_0 = pcd_t.point['f_dc_0'].numpy()
            dc_1 = pcd_t.point['f_dc_1'].numpy()
            dc_2 = pcd_t.point['f_dc_2'].numpy()
            features_dc = np.stack([dc_0, dc_1, dc_2], axis=-1)
            features_dc = np.squeeze(features_dc)
            if features_dc.ndim == 3 and features_dc.shape[1] == 1:
                features_dc = features_dc.squeeze(1)
            
            # Load f_rest_ fields for higher order SH (degrees 1-3)
            f_rest_fields = []
            for i in range(45):  # 45 f_rest_ fields for degree 3
                field_name = f'f_rest_{i}'
                if field_name in pcd_t.point:
                    f_rest_fields.append(field_name)
            
            print(f"Found {len(f_rest_fields)} f_rest_ fields for higher order SH")
            
            if f_rest_fields:
                # Load all f_rest_ fields
                features_rest = np.zeros((len(positions), len(f_rest_fields)))
                for idx, field_name in enumerate(f_rest_fields):
                    data = pcd_t.point[field_name].numpy()
                    features_rest[:, idx] = np.squeeze(data)
                
                # Combine DC and rest terms
                sh_coeffs = np.concatenate([features_dc, features_rest], axis=-1)
                # Determine the SH degree based on number of coefficients
                expected_coeffs = [3, 12, 27, 48]  # For degrees 0, 1, 2, 3
                degrees = [0, 1, 2, 3]
                sh_degree = 0
                for i, expected in enumerate(expected_coeffs):
                    if sh_coeffs.shape[1] >= expected:
                        sh_degree = degrees[i]
                
                print(f"Total SH coefficients: {sh_coeffs.shape[1]} (supports up to degree {sh_degree})")
            else:
                # Only DC terms available
                sh_coeffs = features_dc
                print("Only DC terms available, using degree 0 SH")
                
        else:
            print("No spherical harmonics found, using white")
            # Create default SH coefficients (only DC term)
            sh_coeffs = np.ones((len(positions), 3)) * 0.28209479177387814
        
        # Convert DC term to colors for display (just the first 3 coefficients)
        colors = 0.5 + sh_coeffs[:, :3] * 0.28209479177387814
        colors = np.clip(colors, 0, 1)
        
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
            # PLY stores as (w,x,y,z) in rot_0-3, use directly like working method
            rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1)  # Direct order (w,x,y,z)
            rotations = np.squeeze(rotations)
            if rotations.ndim == 3 and rotations.shape[1] == 1:
                rotations = rotations.squeeze(1)
            
            # Fix quaternion coordinate system to match working method
            # Apply the correct transformation: negate w and x components
            rotations[:, 0] = -rotations[:, 0]  # negate w
            rotations[:, 1] = -rotations[:, 1]  # negate x
            print("Applied quaternion fix: negating w+x components")
            
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
        self.sh_coeffs = sh_coeffs.astype(np.float32)
        self.sh_dim = sh_coeffs.shape[1]
        
        # Determine SH degree based on number of coefficients
        if self.sh_dim == 3:
            sh_degree = 0
        elif self.sh_dim == 12:
            sh_degree = 1
        elif self.sh_dim == 27:
            sh_degree = 2
        elif self.sh_dim == 48:
            sh_degree = 3
        else:
            sh_degree = 0  # Default to degree 0
            
        print(f"Loaded {self.num_points} gaussians with {self.sh_dim} SH coefficients (degree {sh_degree})")
        print(f"Higher order SH support: {'Available' if self.sh_dim > 3 else 'Not available (DC only)'}")
        
        # Create storage buffers for rendering
        self._create_gaussian_data_buffer()
        self._create_sorted_indices_buffer()
        
    
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
        
        # Create SH coefficients (only DC term for naive gaussians)
        self.sh_coeffs = (gau_c - 0.5) / 0.28209479177387814
        self.sh_dim = 3
        
        # Opacity: all fully opaque
        self.opacities = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        # Create storage buffers for rendering
        self._create_gaussian_data_buffer()
        self._create_sorted_indices_buffer()
    
    def render(self, mv_matrix, p_matrix):
        """Render using storage buffer approach like working method"""
        
        # Sort gaussians by depth (like working method)
        self._sort_gaussians(mv_matrix)
        
        # Get viewport dimensions for rendering
        viewport = glGetIntegerv(GL_VIEWPORT)
        viewport_width = float(viewport[2])
        viewport_height = float(viewport[3])
        
        # Set up rendering if not done
        if self.instanced_vao is None:
            self._setup_storage_buffer_rendering()
        
        # Render using storage buffers
        self._render_storage_buffers(viewport_width, viewport_height, mv_matrix, p_matrix)
    
    def _sort_gaussians(self, view_matrix):
        """Sort gaussians by depth using view matrix (like working method)"""
        # Transform positions to view space
        view_matrix_3x4 = view_matrix[:3, :]  # Extract 3x4 part
        
        # Apply view transform: view_space = view_matrix * [pos, 1]
        homogeneous_pos = np.ones((self.num_points, 4), dtype=np.float32)
        homogeneous_pos[:, :3] = self.original_positions
        
        # Transform to view space
        view_space = (view_matrix_3x4 @ homogeneous_pos.T).T  # Shape: (num_points, 3)
        depth = view_space[:, 2]  # Z coordinate in view space
        
        # Sort by depth (farthest to closest for proper alpha blending)
        sorted_indices = np.argsort(depth).astype(np.int32)
        
        # Update sorted indices buffer
        self.sorted_indices = sorted_indices
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.gaussian_order_buffer)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.sorted_indices.nbytes, self.sorted_indices)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    
    
    def _setup_storage_buffer_rendering(self):
        """Set up OpenGL objects for storage buffer rendering (like working method)"""
        if self.instanced_vao is not None:
            return  # Already set up
            
        print("Setting up storage buffer rendering...")
        
        # Create base quad vertices (2D positions) - match working method order
        base_quad = np.array([
            [-1.0,  1.0],  # Top-left     (vertex 0)
            [ 1.0,  1.0],  # Top-right    (vertex 1) 
            [ 1.0, -1.0],  # Bottom-right (vertex 2)
            [-1.0, -1.0],  # Bottom-left  (vertex 3)
        ], dtype=np.float32).flatten()
        
        # Create indices for two triangles - match working method order
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # Create VAO and VBOs
        self.instanced_vao = glGenVertexArrays(1)
        self.base_quad_vbo = glGenBuffers(1)
        self.base_quad_ibo = glGenBuffers(1)
        
        glBindVertexArray(self.instanced_vao)
        
        # Set up base quad vertices (only vertex attribute needed)
        glBindBuffer(GL_ARRAY_BUFFER, self.base_quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, base_quad.nbytes, base_quad, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Set up index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.base_quad_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        print(f"Storage buffer rendering setup complete")
    
    
    def _render_storage_buffers(self, viewport_width, viewport_height, mv_matrix, p_matrix):
        """Render using storage buffers (like working method)"""
        # Enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable depth testing but disable depth writing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_FALSE)
        
        # Use shader program
        glUseProgram(self.instanced_shader_program)
        
        # Calculate hfovxy_focal values to match working method
        fovy_radians = math.radians(45.0)
        aspect_ratio = viewport_width / viewport_height
        tan_half_fovy = math.tan(fovy_radians * 0.5)
        tan_half_fovx = tan_half_fovy * aspect_ratio
        # Working method uses focal length in a different way
        focal = viewport_height / (2.0 * tan_half_fovy)
        hfovxy_focal = [tan_half_fovx, tan_half_fovy, focal]
        
        # Camera position (extract from view matrix)
        view_inv = np.linalg.inv(mv_matrix)
        cam_pos = view_inv[:3, 3]
        
        # Set uniforms (matching working method) - OpenGL expects column-major
        glUniformMatrix4fv(self.instanced_view_matrix_uniform, 1, GL_FALSE, mv_matrix.T.flatten())
        glUniformMatrix4fv(self.instanced_projection_matrix_uniform, 1, GL_FALSE, p_matrix.T.flatten())
        glUniform3fv(self.instanced_hfovxy_focal_uniform, 1, hfovxy_focal)
        glUniform3fv(self.instanced_cam_pos_uniform, 1, cam_pos)
        
        # Debug matrices (only occasionally)
        import time
        if not hasattr(self, '_last_debug_time'):
            self._last_debug_time = 0
        if time.time() - self._last_debug_time > 2.0:  # Every 2 seconds
            print(f"Debug: Camera position: {cam_pos}, focal: {focal:.2f}")
            print(f"Debug: SH mode: {self.sh_render_mode}, SH dim: {self.sh_dim}, render_mod uniform: {self.sh_render_mode}")
            self._last_debug_time = time.time()
        
        glUniform1f(self.instanced_scale_modifier_uniform, 1.0)
        glUniform1i(self.instanced_render_mod_uniform, self.sh_render_mode)  # Use SH render mode
        glUniform1i(self.instanced_sh_dim_uniform, self.sh_dim)
        
        # Bind storage buffers (they should already be bound to binding points 0 and 1)
        # But ensure they're bound correctly
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.gaussian_data_buffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.gaussian_order_buffer)
        
        # Bind VAO and render instanced (each instance is one gaussian)
        glBindVertexArray(self.instanced_vao)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_points)
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
    
    def set_sh_mode(self, mode):
        """Set spherical harmonics rendering mode
        
        Args:
            mode: 0=DC only, 1=DC+degree1, 2=DC+degree1+2, 3=DC+degree1+2+3
        """
        self.sh_render_mode = max(0, min(mode, 3))
        
        # Determine actual available degree based on SH coefficients
        max_available_degree = 0
        if self.sh_dim >= 12:
            max_available_degree = 1
        if self.sh_dim >= 27:
            max_available_degree = 2  
        if self.sh_dim >= 48:
            max_available_degree = 3
            
        actual_mode = min(self.sh_render_mode, max_available_degree)
        
        if actual_mode != self.sh_render_mode:
            print(f"SH mode {self.sh_render_mode} requested but only degree {max_available_degree} available - using {actual_mode}")
        else:
            print(f"SH mode set to {self.sh_render_mode} (degree 0-{self.sh_render_mode})")
            
        # Now we have full SH evaluation!
        if actual_mode > 0:
            print(f"✅ Full higher order SH evaluation active for degree {actual_mode}!")
            print(f"Debug: render_mod will be set to {self.sh_render_mode}, sh_dim={self.sh_dim}")
            print(f"Debug: Shader conditions - deg1: sh_dim({self.sh_dim}) > 3 && render_mod({self.sh_render_mode}) >= 1 = {self.sh_dim > 3 and self.sh_render_mode >= 1}")
            if self.sh_dim > 12:
                print(f"Debug: Shader conditions - deg2: sh_dim({self.sh_dim}) > 12 && render_mod({self.sh_render_mode}) >= 2 = {self.sh_dim > 12 and self.sh_render_mode >= 2}")
            if self.sh_dim > 27:
                print(f"Debug: Shader conditions - deg3: sh_dim({self.sh_dim}) > 27 && render_mod({self.sh_render_mode}) >= 3 = {self.sh_dim > 27 and self.sh_render_mode >= 3}")
        else:
            print(f"Using DC term only (degree 0)")
    
    def get_sh_mode(self):
        """Get current SH rendering mode"""
        return self.sh_render_mode
    
    def get_sh_dim(self):
        """Get current SH dimension"""
        return self.sh_dim
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.gaussian_data_buffer:
            glDeleteBuffers(1, [self.gaussian_data_buffer])
        if self.gaussian_order_buffer:
            glDeleteBuffers(1, [self.gaussian_order_buffer])
        if self.base_quad_vbo:
            glDeleteBuffers(1, [self.base_quad_vbo])
        if hasattr(self, 'base_quad_ibo') and self.base_quad_ibo:
            glDeleteBuffers(1, [self.base_quad_ibo])
        if self.instanced_vao:
            glDeleteVertexArrays(1, [self.instanced_vao])
        if self.instanced_shader_program:
            glDeleteProgram(self.instanced_shader_program)