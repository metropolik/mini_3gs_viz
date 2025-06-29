import glfw
from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians, pi, tan
import os
import sys
import OpenEXR
import Imath
import ctypes
import open3d as o3d

# Vertex shader for grid
grid_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uMVP;

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# Fragment shader for grid
grid_fragment_shader = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(0.0, 0.8, 0.0, 1.0);  // Green color
}
"""

# Vertex shader for skybox
skybox_vertex_shader = """
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

# Fragment shader for skybox
skybox_fragment_shader = """
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

# Vertex shader for point rendering
point_vertex_shader = """
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

# Fragment shader for point rendering
point_fragment_shader = """
#version 330 core
in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0);
}
"""

class FirstPersonCamera:
    def __init__(self, position=(0, 1.6, 0), yaw=-90.0, pitch=0.0):
        self.position = list(position)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 5.0
        self.sensitivity = 0.1
        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True
        
    def get_view_matrix(self):
        """Get view matrix as numpy array"""
        # Calculate direction vector
        front_x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front_y = sin(radians(self.pitch))
        front_z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        
        # Normalize direction vector
        front = np.array([front_x, front_y, front_z])
        front = front / np.linalg.norm(front)
        
        # Calculate right and up vectors
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(front, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, front)
        
        # Create view matrix (lookAt matrix)
        position = np.array(self.position)
        
        # Build the view matrix
        view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, position)],
            [up[0], up[1], up[2], -np.dot(up, position)],
            [-front[0], -front[1], -front[2], np.dot(front, position)],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return view_matrix
        
    def process_keyboard(self, window, delta_time):
        velocity = self.speed * delta_time
        
        # Calculate front and right vectors
        front_x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front_y = sin(radians(self.pitch))
        front_z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        right_x = sin(radians(self.yaw))
        right_z = -cos(radians(self.yaw))
        
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position[0] += front_x * velocity
            self.position[1] += front_y * velocity
            self.position[2] += front_z * velocity
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position[0] -= front_x * velocity
            self.position[1] -= front_y * velocity
            self.position[2] -= front_z * velocity
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position[0] += right_x * velocity
            self.position[2] += right_z * velocity
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position[0] -= right_x * velocity
            self.position[2] -= right_z * velocity
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.position[1] += velocity
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.position[1] -= velocity
            
    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            
        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        # Constrain pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

def perspective(fovy, aspect, near, far):
    """Create a perspective projection matrix"""
    f = 1.0 / tan(radians(fovy) / 2.0)
    nf = 1.0 / (near - far)
    
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) * nf
    result[2, 3] = 2.0 * far * near * nf
    result[3, 2] = -1.0
    
    return result

def compile_shader(source, shader_type):
    """Compile a shader from source"""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {error}")
    
    return shader

def create_shader_program(vertex_source, fragment_source):
    """Create a shader program from vertex and fragment shader sources"""
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program

def create_grid(size=100, spacing=1.0):
    """Create grid vertex data"""
    vertices = []
    
    # Lines parallel to X axis
    for i in range(-size, size + 1, int(spacing)):
        z = float(i)
        vertices.extend([-size, 0.0, z])  # Start point
        vertices.extend([size, 0.0, z])   # End point
    
    # Lines parallel to Z axis
    for i in range(-size, size + 1, int(spacing)):
        x = float(i)
        vertices.extend([x, 0.0, -size])  # Start point
        vertices.extend([x, 0.0, size])   # End point
    
    return np.array(vertices, dtype=np.float32)

def load_exr_texture(filename):
    """Load EXR file as OpenGL texture"""
    cache_file = filename.replace('.exr', '_cache.npz')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached texture from {cache_file}")
        cache = np.load(cache_file)
        texture_data = cache['texture_data']
        width = int(cache['width'])
        height = int(cache['height'])
    else:
        print(f"Loading EXR file {filename}")
        # Open EXR file
        exr_file = OpenEXR.InputFile(filename)
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
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    # Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, texture_data)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    return texture_id

def create_skybox_sphere(segments=32):
    """Create sphere geometry for skybox"""
    vertices = []
    radius = 500.0
    
    # Generate sphere vertices with texture coordinates
    for lat in range(segments):
        lat0 = pi * (-0.5 + float(lat) / segments)
        lat1 = pi * (-0.5 + float(lat + 1) / segments)
        
        for lon in range(segments):
            lon_angle0 = 2 * pi * float(lon) / segments
            lon_angle1 = 2 * pi * float(lon + 1) / segments
            
            # Calculate sphere vertices and texture coordinates
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
            
            x3 = cos(lat1) * cos(lon_angle1)
            y3 = sin(lat1)
            z3 = cos(lat1) * sin(lon_angle1)
            u3 = float(lon + 1) / segments
            v3 = 1.0 - (0.5 + lat1 / pi)
            
            # First triangle
            vertices.extend([x0 * radius, y0 * radius, z0 * radius, u0, v0])
            vertices.extend([x1 * radius, y1 * radius, z1 * radius, u1, v1])
            vertices.extend([x2 * radius, y2 * radius, z2 * radius, u2, v2])
            
            # Second triangle
            vertices.extend([x1 * radius, y1 * radius, z1 * radius, u1, v1])
            vertices.extend([x3 * radius, y3 * radius, z3 * radius, u3, v3])
            vertices.extend([x2 * radius, y2 * radius, z2 * radius, u2, v2])
    
    return np.array(vertices, dtype=np.float32)

def load_ply(ply_path):
    """Load PLY file and extract positions and colors"""
    print(f"Loading Gaussian splat model from {ply_path}")
    
    # Load PLY file using Open3D tensor API
    pcd_t = o3d.t.io.read_point_cloud(ply_path)
    
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
    
    return vertex_data.flatten(), len(positions)

def main():
    # Parse command line arguments
    ply_path = None
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
        if not os.path.exists(ply_path):
            print(f"Error: PLY file '{ply_path}' not found")
            return
    
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Request OpenGL 3.3 core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # Create window
    width, height = 3600, 2000
    
    # Get primary monitor for centering
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    
    # Calculate center position
    pos_x = (mode.size.width - width) // 2
    pos_y = (mode.size.height - height) // 2
    
    window = glfw.create_window(width, height, "First Person Camera Demo", None, None)
    if not window:
        glfw.terminate()
        return
    
    # Set window position to center
    glfw.set_window_pos(window, pos_x, pos_y)
    
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    # Set up OpenGL
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    # Create camera
    camera = FirstPersonCamera()
    
    # Create shader program for grid
    grid_shader = create_shader_program(grid_vertex_shader, grid_fragment_shader)
    mvp_uniform = glGetUniformLocation(grid_shader, "uMVP")
    
    # Create grid VAO and VBO
    grid_vertices = create_grid()
    num_grid_vertices = len(grid_vertices) // 3
    
    grid_vao = glGenVertexArrays(1)
    grid_vbo = glGenBuffers(1)
    
    glBindVertexArray(grid_vao)
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo)
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices, GL_STATIC_DRAW)
    
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(0)
    
    glBindVertexArray(0)
    
    # Create point renderer if PLY file is provided
    point_vao = None
    point_vbo = None
    point_shader = None
    point_mvp_uniform = None
    num_points = 0
    
    if ply_path:
        print("Setting up point renderer...")
        
        # Create point shader program
        point_shader = create_shader_program(point_vertex_shader, point_fragment_shader)
        point_mvp_uniform = glGetUniformLocation(point_shader, "uMVP")
        
        # Load PLY data
        point_vertices, num_points = load_ply(ply_path)
        
        # Create point VAO and VBO
        point_vao = glGenVertexArrays(1)
        point_vbo = glGenBuffers(1)
        
        glBindVertexArray(point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, point_vbo)
        glBufferData(GL_ARRAY_BUFFER, point_vertices.nbytes, point_vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        
        # Enable point size control
        glEnable(GL_PROGRAM_POINT_SIZE)
    
    # Create skybox if background.exr exists
    skybox_vao = None
    skybox_vbo = None
    skybox_shader = None
    skybox_texture = None
    skybox_mvp_uniform = None
    skybox_texture_uniform = None
    num_skybox_vertices = 0
    
    if os.path.exists('background.exr'):
        print("Setting up skybox...")
        
        # Create skybox shader program
        skybox_shader = create_shader_program(skybox_vertex_shader, skybox_fragment_shader)
        skybox_mvp_uniform = glGetUniformLocation(skybox_shader, "uMVP")
        skybox_texture_uniform = glGetUniformLocation(skybox_shader, "uTexture")
        
        # Load skybox texture
        skybox_texture = load_exr_texture('background.exr')
        
        # Create skybox geometry
        skybox_vertices = create_skybox_sphere()
        num_skybox_vertices = len(skybox_vertices) // 5  # 5 floats per vertex (3 pos + 2 tex)
        
        # Create skybox VAO and VBO
        skybox_vao = glGenVertexArrays(1)
        skybox_vbo = glGenBuffers(1)
        
        glBindVertexArray(skybox_vao)
        glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
        glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    # Mouse callback
    def mouse_callback(window, xpos, ypos):
        camera.process_mouse(xpos, ypos)
    
    # Scroll callback for speed adjustment
    def scroll_callback(window, xoffset, yoffset):
        camera.speed *= 1.1 ** yoffset
        camera.speed = max(0.1, min(camera.speed, 50.0))
        print(f"Movement speed: {camera.speed:.1f}")
    
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    # Timing
    last_frame = 0.0
    
    # Main loop
    while not glfw.window_should_close(window):
        # Calculate delta time
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        
        # Process input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        camera.process_keyboard(window, delta_time)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Calculate matrices
        view_matrix = camera.get_view_matrix()
        projection_matrix = perspective(45.0, width/height, 0.1, 1000.0)
        
        # Draw skybox first (with depth write disabled)
        if skybox_shader and skybox_vao:
            glDepthMask(GL_FALSE)  # Disable depth writing
            
            # Remove translation from view matrix for skybox
            view_no_translation = view_matrix.copy()
            view_no_translation[0:3, 3] = 0  # Remove translation
            skybox_mvp = projection_matrix @ view_no_translation
            
            glUseProgram(skybox_shader)
            glUniformMatrix4fv(skybox_mvp_uniform, 1, GL_TRUE, skybox_mvp)
            
            # Bind texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, skybox_texture)
            glUniform1i(skybox_texture_uniform, 0)
            
            glBindVertexArray(skybox_vao)
            glDrawArrays(GL_TRIANGLES, 0, num_skybox_vertices)
            glBindVertexArray(0)
            
            glUseProgram(0)
            glDepthMask(GL_TRUE)  # Re-enable depth writing
        
        # Draw grid
        mvp_matrix = projection_matrix @ view_matrix  # Model is identity
        
        glUseProgram(grid_shader)
        glUniformMatrix4fv(mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, num_grid_vertices)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Draw points
        if point_shader and point_vao:
            glUseProgram(point_shader)
            glUniformMatrix4fv(point_mvp_uniform, 1, GL_TRUE, mvp_matrix)
            
            glBindVertexArray(point_vao)
            glDrawArrays(GL_POINTS, 0, num_points)
            glBindVertexArray(0)
            
            glUseProgram(0)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Cleanup
    glDeleteVertexArrays(1, [grid_vao])
    glDeleteBuffers(1, [grid_vbo])
    glDeleteProgram(grid_shader)
    
    if skybox_vao:
        glDeleteVertexArrays(1, [skybox_vao])
        glDeleteBuffers(1, [skybox_vbo])
        glDeleteProgram(skybox_shader)
        if skybox_texture:
            glDeleteTextures(skybox_texture)
    
    if point_vao:
        glDeleteVertexArrays(1, [point_vao])
        glDeleteBuffers(1, [point_vbo])
        glDeleteProgram(point_shader)
    
    glfw.terminate()

if __name__ == "__main__":
    main()