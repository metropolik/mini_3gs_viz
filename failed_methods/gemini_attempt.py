print("Gemini")
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm

# --- Shaders ---
# The vertex shader is now a simple "pass-through" shader.
# It receives coordinates that are already transformed into clip space.
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec4 aPos; // Input is now a vec4
void main()
{
    gl_Position = aPos; // Directly assign the pre-transformed position
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
}
"""

# --- Camera Class (Unchanged) ---
class Camera:
    def __init__(self, position=glm.vec3(0, 0, 3), front=glm.vec3(0, 0, -1), up=glm.vec3(0, 1, 0)):
        self.position = position
        self.front = front
        self.up = up
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 2.5
        self.sensitivity = 0.1
        self.last_x, self.last_y = 800 / 2, 600 / 2
        self.first_mouse = True

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def process_input(self, dt, keys, mouse_rel):
        # Keyboard movement
        velocity = self.speed * dt
        if keys[K_w]:
            self.position += self.front * velocity
        if keys[K_s]:
            self.position -= self.front * velocity
        if keys[K_a]:
            self.position -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if keys[K_d]:
            self.position += glm.normalize(glm.cross(self.front, self.up)) * velocity

        # Mouse movement
        if self.first_mouse:
            self.last_x, self.last_y = pygame.mouse.get_pos()
            self.first_mouse = False

        xoffset = mouse_rel[0]
        yoffset = -mouse_rel[1] # Reversed

        xoffset *= self.sensitivity
        yoffset *= self.sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        front = glm.vec3()
        front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        front.y = glm.sin(glm.radians(self.pitch))
        front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)

def main():
    print("Pygame init")
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    print("set up pygame")

    glEnable(GL_DEPTH_TEST)

    # Compile shaders
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    # Generate random points in model space
    num_points = 500
    points_model_space = np.random.rand(num_points, 3) * 10 - 5
    # Convert to homogeneous coordinates (add a 1.0 as the w-component)
    points_homogeneous = np.hstack([points_model_space, np.ones((num_points, 1))])

    # Create VBO and VAO
    VBO = glGenBuffers(1)
    VAO = glGenVertexArrays(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    # Use GL_DYNAMIC_DRAW because we will be updating the VBO data every frame
    glBufferData(GL_ARRAY_BUFFER, points_homogeneous.nbytes, points_homogeneous, GL_DYNAMIC_DRAW)

    # The vertex attribute is now a vec4
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    camera = Camera()
    clock = pygame.time.Clock()
    dt = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        # Input
        keys = pygame.key.get_pressed()
        mouse_rel = pygame.mouse.get_rel()
        camera.process_input(dt, keys, mouse_rel)

        # Clear screen
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --- CPU-Side Transformation ---
        # 1. Create MVP matrices
        model = glm.mat4(1.0)
        view = camera.get_view_matrix()
        projection = glm.perspective(glm.radians(45.0), display[0] / display[1], 0.1, 100.0)
        mvp = projection * view * model
        mvp_np = np.array(mvp, dtype=np.float32)

        # 2. Transform vertices on the CPU using numpy
        # The MVP matrix is column-major, so we transpose it for multiplication with row-major numpy arrays.
        transformed_points = points_homogeneous @ mvp_np.T

        # 3. Update the VBO with the new, transformed vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, transformed_points.nbytes, transformed_points)
        
        # 4. Draw points
        glDrawArrays(GL_POINTS, 0, num_points)

        pygame.display.flip()
        dt = clock.tick(60) / 1000.0

    pygame.quit()
    quit()

print("hello")
if __name__ == "__main__":
    print('hello2')
    main()