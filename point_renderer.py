from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians
import os
import open3d as o3d
import ctypes

class PointRenderer:
    def __init__(self, ply_path=None):
        """Initialize point renderer with optional PLY file"""
        self.ply_path = ply_path
        
        # Shader source code
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        
        uniform mat4 uMVP;
        
        out vec3 color;
        
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
            gl_PointSize = 3.0;
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
        self.mvp_uniform = None
        self.num_points = 0
        
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
        
        # Get uniform location
        self.mvp_uniform = glGetUniformLocation(self.shader_program, "uMVP")
    
    def _load_ply(self):
        """Load PLY file and extract positions and colors"""
        print(f"Loading Gaussian splat model from {self.ply_path}")
        
        # Load PLY file using Open3D tensor API
        pcd_t = o3d.t.io.read_point_cloud(self.ply_path)
        
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
        
        # Create interleaved vertex data (position + color)
        vertex_data = np.zeros((len(positions), 6), dtype=np.float32)
        vertex_data[:, :3] = positions
        vertex_data[:, 3:6] = colors
        vertex_data = vertex_data.flatten()
        
        self.num_points = len(positions)
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def render(self, mvp_matrix):
        """Render the points with given MVP matrix"""
        if not self.shader_program or not self.vao:
            return
        
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
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