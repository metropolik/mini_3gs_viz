import glfw
from OpenGL.GL import *
import numpy as np
import math
import random

# Vertex shader source code
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 mvp;

out vec3 color;

void main()
{
    gl_Position = mvp * vec4(aPos, 1.0);
    color = aColor;
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core
in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0);
}
"""

class FirstPersonCamera:
    def __init__(self, position=[0, 0, 3], yaw=-90, pitch=0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 2.5
        self.sensitivity = 0.1
        self.update_vectors()
    
    def update_vectors(self):
        # Calculate front vector
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        self.front = front / np.linalg.norm(front)
        
        # Calculate right and up vectors
        world_up = np.array([0, 1, 0])
        self.right = np.cross(self.front, world_up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)
    
    def get_view_matrix(self):
        return look_at(self.position, self.position + self.front, self.up)
    
    def process_keyboard(self, direction, delta_time):
        velocity = self.speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        elif direction == "BACKWARD":
            self.position -= self.front * velocity
        elif direction == "LEFT":
            self.position -= self.right * velocity
        elif direction == "RIGHT":
            self.position += self.right * velocity
    
    def process_mouse(self, xoffset, yoffset):
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        # Constrain pitch
        self.pitch = max(-89, min(89, self.pitch))
        
        self.update_vectors()

def look_at(eye, center, up):
    """Create a view matrix using eye position, target, and up vector"""
    f = center - eye
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    
    u = np.cross(s, f)
    
    result = np.eye(4, dtype=np.float32)
    result[0, 0] = s[0]
    result[1, 0] = s[1]
    result[2, 0] = s[2]
    result[0, 1] = u[0]
    result[1, 1] = u[1]
    result[2, 1] = u[2]
    result[0, 2] = -f[0]
    result[1, 2] = -f[1]
    result[2, 2] = -f[2]
    result[0, 3] = -np.dot(s, eye)
    result[1, 3] = -np.dot(u, eye)
    result[2, 3] = np.dot(f, eye)
    
    return result

def perspective(fovy, aspect, near, far):
    """Create a perspective projection matrix"""
    f = 1.0 / math.tan(math.radians(fovy) / 2.0)
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

def generate_random_points(num_points, range_x=10, range_y=10, range_z=10):
    """Generate random colored points in 3D space"""
    vertices = []
    for _ in range(num_points):
        # Position
        x = random.uniform(-range_x, range_x)
        y = random.uniform(-range_y, range_y)
        z = random.uniform(-range_z, range_z)
        
        # Color
        r = random.random()
        g = random.random()
        b = random.random()
        
        vertices.extend([x, y, z, r, g, b])
    
    return np.array(vertices, dtype=np.float32)

# Global variables for mouse handling
first_mouse = True
last_x = 400
last_y = 300
camera = None

def mouse_callback(window, xpos, ypos):
    global first_mouse, last_x, last_y, camera
    
    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False
    
    xoffset = xpos - last_x
    yoffset = last_y - ypos  # Reversed since y-coordinates range from bottom to top
    
    last_x = xpos
    last_y = ypos
    
    camera.process_mouse(xoffset, yoffset)

def main():
    global camera
    
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create window
    width, height = 800, 600
    window = glfw.create_window(width, height, "PyOpenGL First Person Camera", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    # Enable depth testing and point size
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_PROGRAM_POINT_SIZE)
    
    # Create shader program
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    
    # Generate random points
    vertices = generate_random_points(500)
    
    # Create VBO and VAO
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    
    # Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, 
                         ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    
    # Initialize camera
    camera = FirstPersonCamera([0, 0, 5])
    
    # Get uniform location
    mvp_loc = glGetUniformLocation(shader_program, "mvp")
    
    # Timing
    delta_time = 0.0
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
        
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            camera.process_keyboard("FORWARD", delta_time)
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            camera.process_keyboard("BACKWARD", delta_time)
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            camera.process_keyboard("LEFT", delta_time)
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            camera.process_keyboard("RIGHT", delta_time)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Use shader program
        glUseProgram(shader_program)
        
        # Calculate MVP matrix
        model = np.eye(4, dtype=np.float32)
        view = camera.get_view_matrix()
        projection = perspective(45.0, width/height, 0.1, 100.0)
        
        # Multiply matrices: MVP = Projection * View * Model
        mvp = np.dot(projection, np.dot(view, model))
        
        # Set uniform
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)  # GL_TRUE to transpose
        
        # Draw points
        glBindVertexArray(vao)
        glPointSize(5.0)
        glDrawArrays(GL_POINTS, 0, len(vertices) // 6)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Cleanup
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)
    
    glfw.terminate()

if __name__ == "__main__":
    main()