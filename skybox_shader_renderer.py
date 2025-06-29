import numpy as np
from OpenGL.GL import *
import ctypes
from math import pi, sin, cos
import OpenEXR
import Imath
import os

class SkyboxShaderRenderer:
    def __init__(self, texture_path='background.exr'):
        self.texture_path = texture_path
        self.texture_id = None
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.mvp_uniform = None
        self.texture_uniform = None
        
        if os.path.exists(texture_path):
            self.load_texture()
        self.setup_shaders()
        self.setup_buffers()
    
    def load_texture(self):
        """Load EXR texture for skybox"""
        cache_file = self.texture_path.replace('.exr', '_cache.npz')
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading cached skybox texture from {cache_file}")
            cache = np.load(cache_file)
            texture_data = cache['texture_data']
            width = cache['width']
            height = cache['height']
        else:
            print(f"Loading EXR skybox file {self.texture_path}")
            # Open EXR file
            exr_file = OpenEXR.InputFile(self.texture_path)
            header = exr_file.header()
            
            # Get dimensions
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Read channels
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = exr_file.channel('R', FLOAT)
            g_str = exr_file.channel('G', FLOAT)
            b_str = exr_file.channel('B', FLOAT)
            
            # Convert to numpy arrays efficiently
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
            
            # Stack into RGB texture
            texture_data = np.stack([r, g, b], axis=2)
            
            # Save cache
            print(f"Saving skybox cache to {cache_file}")
            np.savez_compressed(cache_file, texture_data=texture_data, width=width, height=height)
        
        # Create OpenGL texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, texture_data)
        
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def setup_shaders(self):
        """Create and compile shaders for skybox rendering"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        uniform mat4 uMVP;
        
        out vec2 TexCoord;
        
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform sampler2D uTexture;
        
        void main()
        {
            FragColor = texture(uTexture, TexCoord);
        }
        """
        
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)
        
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Skybox vertex shader compilation failed: {error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)
        
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Skybox fragment shader compilation failed: {error}")
        
        # Create shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Skybox program linking failed: {error}")
        
        # Get uniform locations
        self.mvp_uniform = glGetUniformLocation(self.shader_program, "uMVP")
        self.texture_uniform = glGetUniformLocation(self.shader_program, "uTexture")
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
    
    def setup_buffers(self):
        """Setup VAO and VBO for skybox sphere"""
        vertices = []
        segments = 64
        radius = 500.0
        
        # Generate sphere vertices with texture coordinates
        for lat in range(segments):
            lat0 = pi * (-0.5 + float(lat) / segments)
            lat1 = pi * (-0.5 + float(lat + 1) / segments)
            
            for lon in range(segments):
                lon_angle0 = 2 * pi * float(lon) / segments
                lon_angle1 = 2 * pi * float(lon + 1) / segments
                
                # First triangle
                x0 = cos(lat0) * cos(lon_angle0)
                y0 = sin(lat0)
                z0 = cos(lat0) * sin(lon_angle0)
                u0 = float(lon) / segments
                v0 = 1.0 - (0.5 + lat0 / pi)
                
                x1 = cos(lat1) * cos(lon_angle0)
                y1 = sin(lat1)
                z1 = cos(lat1) * sin(lon_angle0)
                u1 = float(lon) / segments
                v1 = 1.0 - (0.5 + lat1 / pi)
                
                x2 = cos(lat0) * cos(lon_angle1)
                y2 = sin(lat0)
                z2 = cos(lat0) * sin(lon_angle1)
                u2 = float(lon + 1) / segments
                v2 = 1.0 - (0.5 + lat0 / pi)
                
                # Add first triangle
                vertices.extend([x0 * radius, y0 * radius, z0 * radius, u0, v0])
                vertices.extend([x1 * radius, y1 * radius, z1 * radius, u1, v1])
                vertices.extend([x2 * radius, y2 * radius, z2 * radius, u2, v2])
                
                # Second triangle
                x3 = cos(lat1) * cos(lon_angle1)
                y3 = sin(lat1)
                z3 = cos(lat1) * sin(lon_angle1)
                u3 = float(lon + 1) / segments
                v3 = 1.0 - (0.5 + lat1 / pi)
                
                vertices.extend([x1 * radius, y1 * radius, z1 * radius, u1, v1])
                vertices.extend([x3 * radius, y3 * radius, z3 * radius, u3, v3])
                vertices.extend([x2 * radius, y2 * radius, z2 * radius, u2, v2])
        
        vertex_data = np.array(vertices, dtype=np.float32)
        self.num_vertices = len(vertex_data) // 5  # 5 floats per vertex (3 pos + 2 tex)
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def render(self, view_matrix, projection_matrix):
        """Render the skybox"""
        if self.shader_program is None:
            print("No shader program for skybox")
            return
        if self.texture_id is None:
            print("No texture for skybox") 
            return
        
        # Disable depth writing for skybox
        glDepthMask(GL_FALSE)
        
        # For skybox, remove translation from view matrix (keep only rotation)
        view_no_translation = view_matrix.copy()
        view_no_translation[0:3, 3] = 0  # Remove translation
        
        # Compute MVP matrix
        model_matrix = np.eye(4, dtype=np.float32)
        mvp_matrix = projection_matrix @ view_no_translation @ model_matrix
        
        # Use shader program
        glUseProgram(self.shader_program)
        
        # Set uniforms (transpose for column-major)
        glUniformMatrix4fv(self.mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.texture_uniform, 0)
        
        # Render
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_vertices)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Re-enable depth writing
        glDepthMask(GL_TRUE)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
        if self.texture_id:
            glDeleteTextures(1, [self.texture_id])