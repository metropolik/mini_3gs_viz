from OpenGL.GL import *
import numpy as np

class GridRenderer:
    def __init__(self, size=100, spacing=1.0):
        """Initialize grid renderer with specified size and spacing"""
        self.size = size
        self.spacing = spacing
        
        # Shader source code
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 uMVP;
        
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        
        self.fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        
        void main()
        {
            FragColor = vec4(0.0, 0.8, 0.0, 1.0);  // Green color
        }
        """
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.mvp_uniform = None
        self.num_vertices = 0
        
        # Initialize
        self._setup_shaders()
        self._setup_geometry()
    
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
    
    def _setup_geometry(self):
        """Create grid geometry"""
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
    
    def render(self, mvp_matrix):
        """Render the grid with given MVP matrix"""
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
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