import numpy as np
from OpenGL.GL import *
import ctypes

class GridShaderRenderer:
    def __init__(self, size=100, spacing=1.0):
        self.size = size
        self.spacing = spacing
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.mvp_uniform = None
        
        self.setup_shaders()
        self.setup_buffers()
    
    def setup_shaders(self):
        """Create and compile shaders for grid rendering"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 uMVP;
        
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        
        void main()
        {
            FragColor = vec4(0.0, 0.8, 0.0, 1.0);  // Green color
        }
        """
        
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)
        
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Grid vertex shader compilation failed: {error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)
        
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Grid fragment shader compilation failed: {error}")
        
        # Create shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Grid program linking failed: {error}")
        
        # Get uniform location
        self.mvp_uniform = glGetUniformLocation(self.shader_program, "uMVP")
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
    
    def setup_buffers(self):
        """Setup VAO and VBO for grid lines"""
        vertices = []
        
        # Lines parallel to X axis
        for i in range(-self.size, self.size + 1, int(self.spacing)):
            z = float(i)
            vertices.extend([-self.size, 0.0, z])  # Start point
            vertices.extend([self.size, 0.0, z])   # End point
        
        # Lines parallel to Z axis
        for i in range(-self.size, self.size + 1, int(self.spacing)):
            x = float(i)
            vertices.extend([x, 0.0, -self.size])  # Start point
            vertices.extend([x, 0.0, self.size])   # End point
        
        vertex_data = np.array(vertices, dtype=np.float32)
        self.num_vertices = len(vertex_data) // 3
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        glBindVertexArray(0)
    
    def render(self, view_matrix, projection_matrix):
        """Render the grid"""
        if self.shader_program is None:
            return
        
        # Compute MVP matrix (model is identity for grid)
        model_matrix = np.eye(4, dtype=np.float32)
        mvp_matrix = projection_matrix @ view_matrix @ model_matrix
        
        # Use shader program
        glUseProgram(self.shader_program)
        
        # Set MVP uniform (transpose for column-major)
        glUniformMatrix4fv(self.mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
        # Render
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.num_vertices)
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