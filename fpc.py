import glfw
from OpenGL.GL import *
import numpy as np
from math import sin, cos, radians, tan
import os
import sys
from grid_renderer import GridRenderer
from skybox_renderer import SkyboxRenderer
from point_renderer import PointRenderer


class FirstPersonCamera:
    def __init__(self, position=(0, 1.6, 20), yaw=-90.0, pitch=0.0):
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





def main():
    # Parse command line arguments
    ply_path = None
    
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
        if not os.path.exists(ply_path):
            print(f"Error: PLY file '{ply_path}' not found")
            return
    
    print(f"Usage: python fpc.py [ply_file]")
    print(f"Current: ply_file='{ply_path}'")
    print()
    
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
    
    window = glfw.create_window(width, height, "HaViTo", None, None)
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
    
    # Create grid renderer
    grid_renderer = GridRenderer()
    
    # Create skybox renderer
    skybox_renderer = SkyboxRenderer()
    
    # Create point renderer if PLY file is provided
    point_renderer = PointRenderer(ply_path)
    
    
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
    
    # Print controls
    print("\nControls:")
    print("  1 - Gaussian rendering mode")
    print("  2 - Billboard rendering mode")
    print("  3 - Flat Ball rendering mode") 
    print("  4 - Gaussian Ball rendering mode")
    print("  WASD - Move camera")
    print("  Mouse - Look around")
    print("  Scroll - Adjust speed")
    print("  ESC - Exit\n")
    
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
        
        # Render mode controls
        if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
            point_renderer.set_gaussian_mode()
            print("Switched to Gaussian mode")
        elif glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
            point_renderer.set_billboard_mode()
            print("Switched to Billboard mode")
        elif glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
            point_renderer.set_flat_ball_mode()
            print("Switched to Flat Ball mode")
        elif glfw.get_key(window, glfw.KEY_4) == glfw.PRESS:
            point_renderer.set_gaussian_ball_mode()
            print("Switched to Gaussian Ball mode")
        
        camera.process_keyboard(window, delta_time)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Calculate matrices
        view_matrix = camera.get_view_matrix()
        projection_matrix = perspective(45.0, width/height, 0.1, 1000.0)
        
        # Draw skybox first
        skybox_renderer.render(view_matrix, projection_matrix)
        
        # Draw grid
        mvp_matrix = projection_matrix @ view_matrix  # Model is identity
        grid_renderer.render(mvp_matrix)
        
        # Draw points
        mv_matrix = view_matrix  # Model is identity, so MV = V
        point_renderer.render(mv_matrix, projection_matrix)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Cleanup
    grid_renderer.cleanup()
    skybox_renderer.cleanup()
    point_renderer.cleanup()
    
    glfw.terminate()

if __name__ == "__main__":
    main()