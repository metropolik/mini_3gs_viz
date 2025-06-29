import numpy as np
import open3d as o3d
from OpenGL.GL import *
from math import radians, sin, cos
import cupy as cp
import ctypes

class GaussianSplatShaderRenderer:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.positions = None
        self.colors = None
        self.scales = None
        self.rotations = None
        self.opacities = None
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        
        # CUDA kernel for transformations
        self.transform_kernel = None
        self.load_transform_kernel()
        
        # GPU buffers
        self.positions_gpu = None
        self.transformed_positions_gpu = None
        
        self.load_ply()
        self.setup_shaders()
        self.setup_buffers()
        
    def load_transform_kernel(self):
        """Load and compile the CUDA kernel for point transformations"""
        kernel_code = """
        extern "C" __global__
        void transform_points(
            const float* __restrict__ in_points,  // Input points (x, y, z)
            float* __restrict__ out_points,       // Output points (x, y, z)
            const float* __restrict__ transform,  // 4x4 transformation matrix (column-major)
            const int num_points
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (idx >= num_points) return;
            
            // Load the point
            float x = in_points[idx * 3 + 0];
            float y = in_points[idx * 3 + 1];
            float z = in_points[idx * 3 + 2];
            float w = 1.0f;  // Homogeneous coordinate
            
            // Apply transformation (matrix is column-major)
            float tx = transform[0] * x + transform[4] * y + transform[8]  * z + transform[12] * w;
            float ty = transform[1] * x + transform[5] * y + transform[9]  * z + transform[13] * w;
            float tz = transform[2] * x + transform[6] * y + transform[10] * z + transform[14] * w;
            float tw = transform[3] * x + transform[7] * y + transform[11] * z + transform[15] * w;
            
            // Perspective divide
            if (tw != 0.0f && tw != 1.0f) {
                tx /= tw;
                ty /= tw;
                tz /= tw;
            }
            
            // Store the result
            out_points[idx * 3 + 0] = tx;
            out_points[idx * 3 + 1] = ty;
            out_points[idx * 3 + 2] = tz;
        }
        """
        
        self.transform_kernel = cp.RawKernel(kernel_code, 'transform_points')
        
    def load_ply(self):
        print(f"Loading Gaussian splat model from {self.ply_path}")
        
        # Load PLY file using Open3D tensor API
        pcd_t = o3d.t.io.read_point_cloud(self.ply_path)
        
        # Extract positions (x, y, z)
        if pcd_t.point.positions is not None:
            self.positions = pcd_t.point.positions.numpy()
            # Apply 180-degree rotation around X axis
            rot_angle = radians(180)
            rots = sin(rot_angle)
            rotc = cos(rot_angle)
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, rotc, -rots],
                                [0.0, rots, rotc]])
            self.positions = (rot_mat @ self.positions.T).T
            print(f"Loaded {len(self.positions)} splats")
        
        # Extract colors if available (f_dc_0, f_dc_1, f_dc_2)
        if 'f_dc_0' in pcd_t.point:
            dc_0 = pcd_t.point['f_dc_0'].numpy()
            dc_1 = pcd_t.point['f_dc_1'].numpy()
            dc_2 = pcd_t.point['f_dc_2'].numpy()
            self.colors = np.stack([dc_0, dc_1, dc_2], axis=-1)
            self.colors = np.squeeze(self.colors)
            if self.colors.ndim == 3 and self.colors.shape[1] == 1:
                self.colors = self.colors.squeeze(1)
            # Convert from SH DC term to RGB
            self.colors = 0.5 + self.colors * 0.28209479177387814
            self.colors = np.clip(self.colors, 0, 1)
        else:
            self.colors = np.ones((len(self.positions), 3))
        
        # Upload positions to GPU
        if self.positions is not None:
            self.positions_gpu = cp.asarray(self.positions, dtype=cp.float32)
            self.transformed_positions_gpu = cp.empty_like(self.positions_gpu)
    
    def setup_shaders(self):
        """Create and compile shaders"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        
        out vec3 color;
        
        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            gl_PointSize = 3.0;
            color = aColor;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec3 color;
        out vec4 FragColor;
        
        void main()
        {
            FragColor = vec4(color, 1.0);
        }
        """
        
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)
        
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Vertex shader compilation failed: {error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)
        
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Fragment shader compilation failed: {error}")
        
        # Create shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Program linking failed: {error}")
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
    
    def setup_buffers(self):
        """Setup VAO and VBO"""
        if self.positions is None:
            return
            
        # Create vertex data (interleaved positions and colors)
        num_vertices = len(self.positions)
        vertex_data = np.zeros((num_vertices, 6), dtype=np.float32)
        vertex_data[:, :3] = self.positions
        vertex_data[:, 3:6] = self.colors
        vertex_data = vertex_data.flatten()
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)  # 6 floats * 4 bytes
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def render(self, camera_position, view_matrix, projection_matrix):
        if self.positions is None or self.shader_program is None:
            return
        
        # Compute MVP matrix
        projection_matrix_f32 = projection_matrix.astype(np.float32)
        view_matrix_f32 = view_matrix.astype(np.float32)
        mvp_matrix = projection_matrix_f32 @ view_matrix_f32
        
        # Transform vertices using CUDA
        transform_gpu = cp.asarray(mvp_matrix, dtype=cp.float32)
        
        # Configure kernel launch parameters
        num_points = len(self.positions)
        threads_per_block = 256
        blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        self.transform_kernel((blocks_per_grid,), (threads_per_block,),
                            (self.positions_gpu, self.transformed_positions_gpu,
                             transform_gpu, num_points))
        
        # Copy transformed positions back to CPU
        transformed_positions_cpu = cp.asnumpy(self.transformed_positions_gpu)
        
        # Update vertex buffer with transformed positions
        vertex_data = np.zeros((num_points, 6), dtype=np.float32)
        vertex_data[:, :3] = transformed_positions_cpu
        vertex_data[:, 3:6] = self.colors
        vertex_data = vertex_data.flatten()
        
        # Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        
        # Use shader program and render
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, num_points)
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