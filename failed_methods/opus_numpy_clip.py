import glfw
from OpenGL.GL import *
import numpy as np
import math
import random

# Vertex shader source code - now just passes through pre-transformed vertices
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 color;

void main()
{
    gl_Position = vec4(aPos, 1.0);
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

def transform_vertices(vertices, mvp_matrix):
    """Transform vertices by MVP matrix and perform perspective division"""
    # Extract positions (every 6 floats, take the first 3)
    num_vertices = len(vertices) // 6
    positions = vertices.reshape(num_vertices, 6)[:, :3]
    
    # Add w=1 to make 4D homogeneous coordinates
    positions_4d = np.ones((num_vertices, 4), dtype=np.float32)
    positions_4d[:, :3] = positions
    
    # Transform by MVP matrix (transpose for correct multiplication order)
    transformed = np.dot(positions_4d, mvp_matrix.T)
    
    # Store clip space coordinates and w values before division
    clip_space = transformed.copy()
    w_values = transformed[:, 3]
    
    # Perform frustum clipping in clip space
    # A point is inside the frustum if -w <= x,y,z <= w
    # We need to check all 6 planes of the frustum
    valid_mask = (
        (w_values > 0.001) &  # Near plane (w > 0)
        (clip_space[:, 0] >= -w_values) &  # Left plane
        (clip_space[:, 0] <= w_values) &   # Right plane
        (clip_space[:, 1] >= -w_values) &  # Bottom plane
        (clip_space[:, 1] <= w_values) &   # Top plane
        (clip_space[:, 2] >= -w_values) &  # Near plane in z
        (clip_space[:, 2] <= w_values)     # Far plane
    )
    
    # Perform perspective division only for valid vertices
    # Avoid division by zero
    safe_w = np.where(np.abs(w_values) < 0.00001, 0.00001, w_values)
    
    # Divide x, y, z by w
    transformed[:, 0] /= safe_w
    transformed[:, 1] /= safe_w
    transformed[:, 2] /= safe_w
    
    # Create output array with transformed positions and original colors
    result = np.copy(vertices).reshape(num_vertices, 6)
    result[:, :3] = transformed[:, :3]
    
    return result.flatten(), valid_mask

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
    window = glfw.create_window(width, height, "PyOpenGL CPU MVP Transform", None, None)
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
    
    # Generate random points - store original vertices
    original_vertices = generate_random_points(500)
    num_vertices = len(original_vertices) // 6
    
    # Create VBO and VAO
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # Allocate buffer for maximum size
    glBufferData(GL_ARRAY_BUFFER, original_vertices.nbytes, None, GL_DYNAMIC_DRAW)
    
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * original_vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    
    # Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * original_vertices.itemsize, 
                         ctypes.c_void_p(3 * original_vertices.itemsize))
    glEnableVertexAttribArray(1)
    
    # Initialize camera
    camera = FirstPersonCamera([0, 0, 5])
    
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
        
        # Transform vertices on CPU
        transformed_vertices, valid_mask = transform_vertices(original_vertices, mvp)
        
        # Filter out vertices behind camera
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            # Create filtered vertex data
            filtered_vertices = []
            for idx in valid_indices:
                start = idx * 6
                filtered_vertices.extend(transformed_vertices[start:start+6])
            
            filtered_array = np.array(filtered_vertices, dtype=np.float32)
            
            # Update VBO with transformed vertices
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, filtered_array.nbytes, filtered_array)
            
            # Draw points
            glBindVertexArray(vao)
            glPointSize(5.0)
            glDrawArrays(GL_POINTS, 0, len(filtered_array) // 6)
        
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