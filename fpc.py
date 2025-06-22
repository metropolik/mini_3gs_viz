import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from math import sin, cos, radians, pi, atan2, sqrt
import OpenEXR
import Imath
import array
import os
import sys
from gaussian_splat_renderer import GaussianSplatRenderer

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
        # Calculate direction vector
        front_x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front_y = sin(radians(self.pitch))
        front_z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        
        # Look at point
        center = [
            self.position[0] + front_x,
            self.position[1] + front_y,
            self.position[2] + front_z
        ]
        
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            center[0], center[1], center[2],
            0, 1, 0
        )
        
    def process_keyboard(self, window, delta_time):
        velocity = self.speed * delta_time
        
        # Calculate front and right vectors
        front_x = cos(radians(self.yaw))
        front_z = sin(radians(self.yaw))
        right_x = sin(radians(self.yaw))
        right_z = -cos(radians(self.yaw))
        
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position[0] += front_x * velocity
            self.position[2] += front_z * velocity
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position[0] -= front_x * velocity
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

def draw_grid(size=100, spacing=1.0):
    glBegin(GL_LINES)
    glColor3f(0.0, 0.8, 0.0)  # Green color
    
    # Draw lines parallel to X axis
    for i in range(-size, size + 1, int(spacing)):
        z = float(i)
        glVertex3f(-size, 0, z)
        glVertex3f(size, 0, z)
        
    # Draw lines parallel to Z axis
    for i in range(-size, size + 1, int(spacing)):
        x = float(i)
        glVertex3f(x, 0, -size)
        glVertex3f(x, 0, size)
        
    glEnd()

def load_exr_texture(filename):
    cache_file = filename.replace('.exr', '_cache.npz')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached texture from {cache_file}")
        cache = np.load(cache_file)
        texture_data = cache['texture_data']
        width = cache['width']
        height = cache['height']
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
        
        # Convert to numpy arrays efficiently
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
    
    return texture_id

def draw_skybox(texture_id, segments=64):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor3f(1.0, 1.0, 1.0)
    
    # Draw sphere inside-out for skybox
    radius = 500.0
    
    for lat in range(segments):
        lat0 = pi * (-0.5 + float(lat) / segments)
        lat1 = pi * (-0.5 + float(lat + 1) / segments)
        
        glBegin(GL_QUAD_STRIP)
        for lon in range(segments + 1):
            lon_angle = 2 * pi * float(lon) / segments
            
            # First vertex
            x0 = cos(lat0) * cos(lon_angle)
            y0 = sin(lat0)
            z0 = cos(lat0) * sin(lon_angle)
            
            # Second vertex
            x1 = cos(lat1) * cos(lon_angle)
            y1 = sin(lat1)
            z1 = cos(lat1) * sin(lon_angle)
            
            # Texture coordinates (equirectangular mapping)
            u = float(lon) / segments
            v0 = 1.0 - (0.5 + lat0 / pi)
            v1 = 1.0 - (0.5 + lat1 / pi)
            
            glTexCoord2f(u, v0)
            glVertex3f(x0 * radius, y0 * radius, z0 * radius)
            
            glTexCoord2f(u, v1)
            glVertex3f(x1 * radius, y1 * radius, z1 * radius)
            
        glEnd()
    
    glDisable(GL_TEXTURE_2D)

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
    
    # Set up perspective projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width/height, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create camera
    camera = FirstPersonCamera()
    
    # Load background texture
    background_texture = load_exr_texture('background.exr')
    
    # Load Gaussian splat model if provided
    splat_renderer = None
    if ply_path:
        splat_renderer = GaussianSplatRenderer(ply_path)
    
    # Mouse callback
    def mouse_callback(window, xpos, ypos):
        camera.process_mouse(xpos, ypos)
    
    # Scroll callback for speed adjustment
    def scroll_callback(window, xoffset, yoffset):
        camera.speed *= 1.1 ** yoffset  # Increase/decrease by 10% per scroll
        camera.speed = max(0.1, min(camera.speed, 50.0))  # Clamp between 0.1 and 50
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
        
        # Set view matrix
        glLoadIdentity()
        camera.get_view_matrix()
        
        # Draw skybox first (disable depth writing)
        glDepthMask(GL_FALSE)
        draw_skybox(background_texture)
        glDepthMask(GL_TRUE)
        
        # Draw grid
        draw_grid()
        
        # Draw Gaussian splats if loaded
        if splat_renderer:
            # Get current matrices for proper rendering
            projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
            modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
            splat_renderer.render(camera.position, modelview_matrix, projection_matrix)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()

if __name__ == "__main__":
    main()