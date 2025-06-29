from OpenGL.GL import *
import numpy as np
from math import sin, cos, pi
import os
import OpenEXR
import Imath
import ctypes

class SkyboxRenderer:
    def __init__(self, texture_path='background.exr', segments=32):
        """Initialize skybox renderer with texture and geometry"""
        self.texture_path = texture_path
        self.segments = segments
        
        # Shader source code
        self.vertex_shader_source = """
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
        
        self.fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform sampler2D uTexture;
        
        void main()
        {
            vec3 color = texture(uTexture, TexCoord).rgb;
            FragColor = vec4(color, 1.0);
        }
        """
        
        # OpenGL objects
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.texture_id = None
        self.mvp_uniform = None
        self.texture_uniform = None
        self.num_vertices = 0
        
        # Initialize if texture exists
        if os.path.exists(texture_path):
            self._setup_shaders()
            self._load_texture()
            self._setup_geometry()
        else:
            print(f"Skybox texture '{texture_path}' not found, skybox disabled")
    
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
        print("Setting up skybox shaders...")
        
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
        
        # Get uniform locations
        self.mvp_uniform = glGetUniformLocation(self.shader_program, "uMVP")
        self.texture_uniform = glGetUniformLocation(self.shader_program, "uTexture")
    
    def _load_texture(self):
        """Load EXR file as OpenGL texture"""
        cache_file = self.texture_path.replace('.exr', '_cache.npz')
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading cached texture from {cache_file}")
            cache = np.load(cache_file)
            texture_data = cache['texture_data']
            width = int(cache['width'])
            height = int(cache['height'])
        else:
            print(f"Loading EXR file {self.texture_path}")
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
            
            # Convert to numpy arrays
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
            
            # Stack into RGB texture
            texture_data = np.stack([r, g, b], axis=2)
            
            # Save cache
            print(f"Saving cache to {cache_file}")
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
    
    def _setup_geometry(self):
        """Create sphere geometry for skybox"""
        vertices = []
        radius = 500.0
        
        # Generate sphere vertices with texture coordinates
        for lat in range(self.segments):
            lat0 = pi * (-0.5 + float(lat) / self.segments)
            lat1 = pi * (-0.5 + float(lat + 1) / self.segments)
            
            for lon in range(self.segments):
                lon_angle0 = 2 * pi * float(lon) / self.segments
                lon_angle1 = 2 * pi * float(lon + 1) / self.segments
                
                # Calculate sphere vertices and texture coordinates
                x0 = cos(lat0) * cos(lon_angle0)
                y0 = sin(lat0)
                z0 = cos(lat0) * sin(lon_angle0)
                u0 = float(lon) / self.segments
                v0 = 1.0 - (0.5 + lat0 / pi)
                
                x1 = cos(lat1) * cos(lon_angle0)
                y1 = sin(lat1)
                z1 = cos(lat1) * sin(lon_angle0)
                u1 = float(lon) / self.segments
                v1 = 1.0 - (0.5 + lat1 / pi)
                
                x2 = cos(lat0) * cos(lon_angle1)
                y2 = sin(lat0)
                z2 = cos(lat0) * sin(lon_angle1)
                u2 = float(lon + 1) / self.segments
                v2 = 1.0 - (0.5 + lat0 / pi)
                
                x3 = cos(lat1) * cos(lon_angle1)
                y3 = sin(lat1)
                z3 = cos(lat1) * sin(lon_angle1)
                u3 = float(lon + 1) / self.segments
                v3 = 1.0 - (0.5 + lat1 / pi)
                
                # First triangle
                vertices.extend([x0 * radius, y0 * radius, z0 * radius, u0, v0])
                vertices.extend([x1 * radius, y1 * radius, z1 * radius, u1, v1])
                vertices.extend([x2 * radius, y2 * radius, z2 * radius, u2, v2])
                
                # Second triangle
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
        if not self.shader_program or not self.vao:
            return
        
        # Disable depth writing for skybox
        glDepthMask(GL_FALSE)
        
        # Remove translation from view matrix for skybox
        view_no_translation = view_matrix.copy()
        view_no_translation[0:3, 3] = 0  # Remove translation
        skybox_mvp = projection_matrix @ view_no_translation
        
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_uniform, 1, GL_TRUE, skybox_mvp)
        
        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.texture_uniform, 0)
        
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
            glDeleteTextures(self.texture_id)