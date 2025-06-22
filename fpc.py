import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from math import sin, cos, radians

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

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create window
    width, height = 3600, 2000 
    window = glfw.create_window(width, height, "First Person Camera Demo", None, None)
    if not window:
        glfw.terminate()
        return
    
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
    
    # Mouse callback
    def mouse_callback(window, xpos, ypos):
        camera.process_mouse(xpos, ypos)
    
    glfw.set_cursor_pos_callback(window, mouse_callback)
    
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
        
        # Draw grid
        draw_grid()
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()

if __name__ == "__main__":
    main()