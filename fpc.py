import glfw
from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians, pi, tan
import os
import sys

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
    
    # TODO: Gaussian splat renderer will be added here
    # if ply_path:
    #     splat_renderer = GaussianSplatShaderRenderer(ply_path)
    
    # TODO: Skybox renderer will be added here
    # if os.path.exists('background.exr'):
    #     skybox_renderer = SkyboxShaderRenderer()
    
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
        mvp_matrix = projection_matrix @ view_matrix  # Model is identity
        
        # Draw grid
        glUseProgram(grid_shader)
        glUniformMatrix4fv(mvp_uniform, 1, GL_TRUE, mvp_matrix)
        
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, num_grid_vertices)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # TODO: Draw skybox
        # if skybox_renderer:
        #     skybox_renderer.render(view_matrix, projection_matrix)
        
        # TODO: Draw Gaussian splats
        # if splat_renderer:
        #     splat_renderer.render(camera.position, view_matrix, projection_matrix)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Cleanup
    glDeleteVertexArrays(1, [grid_vao])
    glDeleteBuffers(1, [grid_vbo])
    glDeleteProgram(grid_shader)
    
    glfw.terminate()

if __name__ == "__main__":
    main()