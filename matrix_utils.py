import numpy as np
import math

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