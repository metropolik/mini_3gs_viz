import numpy as np
import math


def transform_points(matrix, input_positions, output_positions, num_points):
    """
    Python implementation of transform_points kernel using NumPy
    
    Args:
        matrix: 4x4 transformation matrix (flattened, row-major order)
        input_positions: Input positions (flattened: x,y,z,w,x,y,z,w,...)
        output_positions: Output positions (flattened: x,y,z,w,x,y,z,w,...)
        num_points: Number of points to transform
    """
    # Reshape matrix from flat array to 4x4
    transform_matrix = matrix.reshape(4, 4)
    
    # Reshape input positions to (num_points, 4)
    points = input_positions.reshape(num_points, 4)
    
    # Apply transformation: result = points @ matrix.T (since matrix is row-major)
    transformed_points = points @ transform_matrix.T
    
    # Flatten and store in output array
    output_positions[:] = transformed_points.flatten()


def transform_points_with_perspective(matrix, input_positions, output_positions, num_points):
    """
    Python implementation of transform_points_with_perspective kernel using NumPy
    
    Args:
        matrix: 4x4 transformation matrix (flattened, row-major order)
        input_positions: Input positions (flattened: x,y,z,w,x,y,z,w,...)
        output_positions: Output positions (flattened: x,y,z,x,y,z,...)
        num_points: Number of points to transform
    """
    # Reshape matrix from flat array to 4x4
    transform_matrix = matrix.reshape(4, 4)
    
    # Reshape input positions to (num_points, 4)
    points = input_positions.reshape(num_points, 4)
    
    # Apply transformation: result = points @ matrix.T
    transformed_points = points @ transform_matrix.T
    
    # Perform perspective division
    w_coords = transformed_points[:, 3]
    # Avoid division by zero
    valid_w = np.where(np.abs(w_coords) >= 1e-8, w_coords, 1e-8)
    
    # Divide x, y, z by w
    perspective_points = transformed_points[:, :3] / valid_w[:, np.newaxis]
    
    # Flatten and store in output array
    output_positions[:] = perspective_points.flatten()


def project_to_ndc(vx, vy, vz, vw, proj_matrix):
    """
    Python implementation of project_to_ndc device function
    
    Args:
        vx, vy, vz, vw: View space coordinates
        proj_matrix: Projection matrix (4x4)
    
    Returns:
        ndc_x, ndc_y, ndc_z: NDC coordinates
    """
    # Apply projection matrix
    clip_x = proj_matrix[0] * vx + proj_matrix[1] * vy + proj_matrix[2] * vz + proj_matrix[3] * vw
    clip_y = proj_matrix[4] * vx + proj_matrix[5] * vy + proj_matrix[6] * vz + proj_matrix[7] * vw
    clip_z = proj_matrix[8] * vx + proj_matrix[9] * vy + proj_matrix[10] * vz + proj_matrix[11] * vw
    clip_w = proj_matrix[12] * vx + proj_matrix[13] * vy + proj_matrix[14] * vz + proj_matrix[15] * vw
    
    # Perform perspective division
    inv_w = 1.0 / (clip_w if abs(clip_w) >= 1e-8 else 1e-8)
    ndc_x = clip_x * inv_w
    ndc_y = clip_y * inv_w
    ndc_z = clip_z * inv_w
    
    return ndc_x, ndc_y, ndc_z


def compute_2d_covariance(view_space_positions, scales, rotations, mv_matrix, proj_matrix,
                         cov2d_data, quad_params, visibility_mask, viewport_width, viewport_height, num_points):
    """
    Python implementation of compute_2d_covariance kernel
    
    Args:
        view_space_positions: View space positions (4 components each)
        scales: Scale vectors (3 components each)
        rotations: Rotation quaternions (4 components each)
        mv_matrix: Model-view matrix (4x4)
        proj_matrix: Projection matrix (4x4)
        cov2d_data: Output 2D covariance matrices (3 components: cov[0,0], cov[0,1], cov[1,1])
        quad_params: Output quad parameters (5 components: center_x, center_y, radius_x, radius_y, ndc_z)
        visibility_mask: Output visibility mask (1 if visible, 0 if culled)
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        num_points: Number of points
    """
    for idx in range(num_points):
        # Load Gaussian parameters
        pos_offset = idx * 4
        scale_offset = idx * 3
        rot_offset = idx * 4
        cov_offset = idx * 3
        quad_offset = idx * 5
        
        # Load view space position
        vx = view_space_positions[pos_offset + 0]
        vy = view_space_positions[pos_offset + 1]
        vz = view_space_positions[pos_offset + 2]
        vw = view_space_positions[pos_offset + 3]
        
        # Don't cull for performance - render all quads
        visibility_mask[idx] = 1  # Mark all as visible
        
        # Load scale and rotation
        sx = scales[scale_offset + 0]
        sy = scales[scale_offset + 1]
        sz = scales[scale_offset + 2]
        
        qw = rotations[rot_offset + 0]
        qx = rotations[rot_offset + 1]
        qy = rotations[rot_offset + 2]
        qz = rotations[rot_offset + 3]
        
        # Build 3D covariance matrix like working method: Σ = (S*R)^T * (S*R)
        # First build rotation matrix R from quaternion
        r11 = 1.0 - 2.0 * (qy*qy + qz*qz)
        r12 = 2.0 * (qx*qy - qw*qz)
        r13 = 2.0 * (qx*qz + qw*qy)
        r21 = 2.0 * (qx*qy + qw*qz)
        r22 = 1.0 - 2.0 * (qx*qx + qz*qz)
        r23 = 2.0 * (qy*qz - qw*qx)
        r31 = 2.0 * (qx*qz - qw*qy)
        r32 = 2.0 * (qy*qz + qw*qx)
        r33 = 1.0 - 2.0 * (qx*qx + qy*qy)
        
        # Compute M = S * R (scale THEN rotate like working method)
        # S is diagonal, so this scales the columns of R
        m11 = sx * r11; m12 = sx * r12; m13 = sx * r13
        m21 = sy * r21; m22 = sy * r22; m23 = sy * r23
        m31 = sz * r31; m32 = sz * r32; m33 = sz * r33
        
        # Compute Σ = M^T * M = (S*R)^T * (S*R)
        cov3d_00 = m11*m11 + m21*m21 + m31*m31
        cov3d_01 = m11*m12 + m21*m22 + m31*m32
        cov3d_02 = m11*m13 + m21*m23 + m31*m33
        cov3d_11 = m12*m12 + m22*m22 + m32*m32
        cov3d_12 = m12*m13 + m22*m23 + m32*m33
        cov3d_22 = m13*m13 + m23*m23 + m33*m33
        
        # Extract the 3x3 rotation part of the model-view matrix
        w00 = mv_matrix[0]; w01 = mv_matrix[1]; w02 = mv_matrix[2]
        w10 = mv_matrix[4]; w11 = mv_matrix[5]; w12 = mv_matrix[6]
        w20 = mv_matrix[8]; w21 = mv_matrix[9]; w22 = mv_matrix[10]
        
        # Extract proper camera intrinsics like working method
        # Working method uses: computeCov2D(pos, focal_x, focal_y, tan_fovx, tan_fovy, cov3d, view)
        
        # From our camera setup: FOV=45°, aspect=width/height
        fovy_radians = math.radians(45.0)
        aspect_ratio = viewport_width / viewport_height
        
        # Calculate half-FOV tangents (hfovxy_focal.xy equivalent)
        tan_half_fovy = math.tan(fovy_radians * 0.5)
        tan_half_fovx = tan_half_fovy * aspect_ratio
        
        # Calculate focal lengths (hfovxy_focal.z equivalent)
        focal_y = 1.0 / tan_half_fovy
        focal_x = focal_y / aspect_ratio
        
        # Use the working method's values directly
        tan_fovx = tan_half_fovx
        tan_fovy = tan_half_fovy
        fx = focal_x  
        fy = focal_y
        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        
        # Clamp the view space position to prevent extreme values
        txtz = vx / vz
        tytz = vy / vz
        txtz = min(limx, max(-limx, txtz))
        tytz = min(limy, max(-limy, tytz))
        
        # Update view space position with clamped values
        clamped_vx = txtz * vz
        clamped_vy = tytz * vz
        
        # Build Jacobian matrix J
        inv_z = 1.0 / abs(vz)
        inv_z2 = inv_z * inv_z
        
        j00 = fx * inv_z
        j01 = 0.0
        j02 = -fx * clamped_vx * inv_z2
        j10 = 0.0
        j11 = fy * inv_z
        j12 = -fy * clamped_vy * inv_z2
        
        # Compute T = W * J (3x3 matrix multiplication)
        # Simplified T matrix (many terms are 0):
        t00 = w00 * j00  # = w00*fx/z
        t01 = w01 * j11  # = w01*fy/z  
        t02 = w00 * j02 + w01 * j12
        t10 = w10 * j00  # = w10*fx/z
        t11 = w11 * j11  # = w11*fy/z
        t12 = w10 * j02 + w11 * j12  
        t20 = w20 * j00  # = w20*fx/z
        t21 = w21 * j11  # = w21*fy/z
        t22 = w20 * j02 + w21 * j12
        
        # Compute 2D covariance: cov2D = T^T * transpose(cov3D) * T
        # First compute transpose(cov3D) * T
        ct00 = cov3d_00*t00 + cov3d_01*t10 + cov3d_02*t20
        ct01 = cov3d_00*t01 + cov3d_01*t11 + cov3d_02*t21
        ct02 = cov3d_00*t02 + cov3d_01*t12 + cov3d_02*t22
        ct10 = cov3d_01*t00 + cov3d_11*t10 + cov3d_12*t20
        ct11 = cov3d_01*t01 + cov3d_11*t11 + cov3d_12*t21
        ct12 = cov3d_01*t02 + cov3d_11*t12 + cov3d_12*t22
        ct20 = cov3d_02*t00 + cov3d_12*t10 + cov3d_22*t20
        ct21 = cov3d_02*t01 + cov3d_12*t11 + cov3d_22*t21
        ct22 = cov3d_02*t02 + cov3d_12*t12 + cov3d_22*t22
        
        # Now compute T^T * (transpose(cov3D) * T) = T^T * CT
        cov2d_00 = t00*ct00 + t10*ct10 + t20*ct20
        cov2d_01 = t00*ct01 + t10*ct11 + t20*ct21
        cov2d_11 = t01*ct01 + t11*ct11 + t21*ct21
        
        # Add regularization term to ensure positive definiteness
        cov2d_00 += 1e-6
        cov2d_11 += 1e-6
        
        # Store 2D covariance (symmetric, so store upper triangle)
        cov2d_data[cov_offset + 0] = cov2d_00
        cov2d_data[cov_offset + 1] = cov2d_01
        cov2d_data[cov_offset + 2] = cov2d_11
        
        # Compute eigenvalues for quad sizing (3σ bounds)
        trace = cov2d_00 + cov2d_11
        det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01
        discriminant = trace * trace - 4.0 * det
        
        if discriminant < 0.0:
            discriminant = 0.0
        
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = 0.5 * (trace + sqrt_disc)
        lambda2 = 0.5 * (trace - sqrt_disc)
        
        # Compute radii (3σ = 3 * sqrt(eigenvalue))
        radius_x = 3.0 * math.sqrt(max(lambda1, 1e-6))
        radius_y = 3.0 * math.sqrt(max(lambda2, 1e-6))
        
        # Ensure minimum size for degenerate cases
        radius_x = max(radius_x, 1e-6)
        radius_y = max(radius_y, 1e-6)
        
        # Project center to normalized device coordinates (NDC)
        ndc_x, ndc_y, ndc_z = project_to_ndc(vx, vy, vz, vw, proj_matrix)
        
        # Handle invalid values by clamping
        if not math.isfinite(ndc_x): ndc_x = 0.0
        if not math.isfinite(ndc_y): ndc_y = 0.0
        if not math.isfinite(ndc_z): ndc_z = 0.0
        
        # The radii are already in the correct space
        radius_x_ndc = radius_x
        radius_y_ndc = radius_y
        
        # Store quad parameters (all in NDC space)
        quad_params[quad_offset + 0] = ndc_x
        quad_params[quad_offset + 1] = ndc_y
        quad_params[quad_offset + 2] = radius_x_ndc
        quad_params[quad_offset + 3] = radius_y_ndc
        quad_params[quad_offset + 4] = ndc_z  # Store NDC z for depth


def compact_visible_quads(quad_vertices_in, quad_uvs_in, quad_data_in, visibility_mask, prefix_sum,
                         quad_vertices_out, quad_uvs_out, quad_data_out, num_points):
    """
    Python implementation of compact_visible_quads kernel
    """
    for idx in range(num_points):
        # Skip if not visible
        if visibility_mask[idx] == 0:
            continue
        
        # Get output index from prefix sum (maintains sorted order)
        output_idx = prefix_sum[idx]
        
        # Copy quad vertices (4 vertices per quad)
        for v in range(4):
            input_vertex_idx = idx * 4 + v
            output_vertex_idx = output_idx * 4 + v
            
            # Copy vertex data (8 floats per vertex: x,y,z,r,g,b,center_x,center_y)
            for f in range(8):
                quad_vertices_out[output_vertex_idx * 8 + f] = quad_vertices_in[input_vertex_idx * 8 + f]
            
            # Copy UV data (2 floats per vertex: u,v)
            for f in range(2):
                quad_uvs_out[output_vertex_idx * 2 + f] = quad_uvs_in[input_vertex_idx * 2 + f]
        
        # Copy quad data (6 floats per quad: opacity, inv_cov components, radii)
        for f in range(6):
            quad_data_out[output_idx * 6 + f] = quad_data_in[idx * 6 + f]


def generate_quad_vertices(quad_params, cov2d_data, visibility_mask, colors, opacities,
                          quad_vertices, quad_uvs, quad_data, visible_count, num_points):
    """
    Python implementation of generate_quad_vertices kernel
    """
    for idx in range(num_points):
        # Load quad parameters
        param_offset = idx * 5
        center_x = quad_params[param_offset + 0]
        center_y = quad_params[param_offset + 1]
        radius_x = quad_params[param_offset + 2]
        radius_y = quad_params[param_offset + 3]
        ndc_z = quad_params[param_offset + 4]
        
        # Use the original sorted index to maintain depth order
        quad_idx = idx
        
        # Load 2D covariance matrix
        cov_offset = idx * 3
        cov_00 = cov2d_data[cov_offset + 0]
        cov_01 = cov2d_data[cov_offset + 1]
        cov_11 = cov2d_data[cov_offset + 2]
        
        # Compute inverse of 2D covariance matrix for fragment shader
        det = cov_00 * cov_11 - cov_01 * cov_01
        
        # Skip if covariance matrix is degenerate
        if det <= 1e-12 or cov_00 <= 1e-12 or cov_11 <= 1e-12:
            continue
        
        inv_det = 1.0 / det
        inv_cov_00 = cov_11 * inv_det
        inv_cov_01 = -cov_01 * inv_det
        inv_cov_11 = cov_00 * inv_det
        
        # Load color and opacity
        color_offset = idx * 3
        r = colors[color_offset + 0]
        g = colors[color_offset + 1]
        b = colors[color_offset + 2]
        opacity = opacities[idx]
        
        # Compute eigenvectors for oriented quad
        trace = cov_00 + cov_11
        discriminant = trace * trace - 4.0 * det
        sqrt_disc = math.sqrt(max(discriminant, 0.0))
        lambda1 = 0.5 * (trace + sqrt_disc)
        
        # Eigenvector corresponding to lambda1
        if abs(cov_01) > 1e-6:
            evec_x = lambda1 - cov_11
            evec_y = cov_01
            norm = math.sqrt(evec_x * evec_x + evec_y * evec_y)
            evec_x /= norm
            evec_y /= norm
        else:
            # Diagonal matrix case
            evec_x = 1.0
            evec_y = 0.0
        
        # Generate 4 vertices for the quad
        offsets_x = [-1.0, 1.0, -1.0, 1.0]
        offsets_y = [-1.0, -1.0, 1.0, 1.0]
        uvs = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        
        for i in range(4):
            vertex_idx = quad_idx * 4 + i
            vertex_offset = vertex_idx * 8  # 8 floats per vertex
            uv_offset = vertex_idx * 2      # 2 floats per vertex
            
            # Rotate and scale offset by covariance eigenvectors
            local_x = offsets_x[i] * radius_x
            local_y = offsets_y[i] * radius_y
            
            world_x = evec_x * local_x - evec_y * local_y
            world_y = evec_y * local_x + evec_x * local_y
            
            # Store vertex position (in NDC space with proper depth)
            quad_vertices[vertex_offset + 0] = center_x + world_x
            quad_vertices[vertex_offset + 1] = center_y + world_y
            quad_vertices[vertex_offset + 2] = ndc_z
            
            # Store vertex color
            quad_vertices[vertex_offset + 3] = r
            quad_vertices[vertex_offset + 4] = g
            quad_vertices[vertex_offset + 5] = b
            
            # Store Gaussian center position in NDC space
            quad_vertices[vertex_offset + 6] = center_x
            quad_vertices[vertex_offset + 7] = center_y
            
            # Store UV coordinates
            quad_uvs[uv_offset + 0] = uvs[i * 2 + 0]
            quad_uvs[uv_offset + 1] = uvs[i * 2 + 1]
        
        # Store per-quad data for fragment shader
        quad_data_offset = quad_idx * 6
        quad_data[quad_data_offset + 0] = opacity
        quad_data[quad_data_offset + 1] = inv_cov_00
        quad_data[quad_data_offset + 2] = inv_cov_01
        quad_data[quad_data_offset + 3] = inv_cov_11
        quad_data[quad_data_offset + 4] = radius_x
        quad_data[quad_data_offset + 5] = radius_y


def generate_quad_indices(indices, num_quads):
    """
    Python implementation of generate_quad_indices kernel
    """
    for idx in range(num_quads):
        # Each quad needs 6 indices (2 triangles)
        base_vertex = idx * 4
        base_index = idx * 6
        
        # First triangle: bottom-left, bottom-right, top-left (0, 1, 2)
        indices[base_index + 0] = base_vertex + 0
        indices[base_index + 1] = base_vertex + 1
        indices[base_index + 2] = base_vertex + 2
        
        # Second triangle: bottom-right, top-right, top-left (1, 3, 2)
        indices[base_index + 3] = base_vertex + 1
        indices[base_index + 4] = base_vertex + 3
        indices[base_index + 5] = base_vertex + 2


def generate_instance_data(quad_params, cov2d_data, visibility_mask, colors, opacities,
                          instance_data, num_points):
    """
    Python implementation of generate_instance_data kernel
    """
    for idx in range(num_points):
        # Load quad parameters
        param_offset = idx * 5
        center_x = quad_params[param_offset + 0]
        center_y = quad_params[param_offset + 1]
        radius_x = quad_params[param_offset + 2]
        radius_y = quad_params[param_offset + 3]
        ndc_z = quad_params[param_offset + 4]
        
        # Load 2D covariance matrix
        cov_offset = idx * 3
        cov_00 = cov2d_data[cov_offset + 0]
        cov_01 = cov2d_data[cov_offset + 1]
        cov_11 = cov2d_data[cov_offset + 2]
        
        # Compute inverse of 2D covariance matrix
        det = cov_00 * cov_11 - cov_01 * cov_01
        if det <= 1e-12:
            det = 1e-12  # Ensure non-zero determinant
        
        inv_det = 1.0 / det
        inv_cov_00 = cov_11 * inv_det
        inv_cov_01 = -cov_01 * inv_det
        inv_cov_11 = cov_00 * inv_det
        
        # Load color and opacity
        color_offset = idx * 3
        r = colors[color_offset + 0]
        g = colors[color_offset + 1]
        b = colors[color_offset + 2]
        opacity = opacities[idx]
        
        # Pack instance data (10 floats per instance)
        instance_offset = idx * 10
        instance_data[instance_offset + 0] = center_x
        instance_data[instance_offset + 1] = center_y
        instance_data[instance_offset + 2] = ndc_z
        instance_data[instance_offset + 3] = r
        instance_data[instance_offset + 4] = g
        instance_data[instance_offset + 5] = b
        instance_data[instance_offset + 6] = opacity
        instance_data[instance_offset + 7] = inv_cov_00
        instance_data[instance_offset + 8] = inv_cov_01
        instance_data[instance_offset + 9] = inv_cov_11