# Gaussian Splatting Renderer - Development Plan

## Current State

### Files and Layout
- **`point_renderer.py`** - Main Gaussian renderer class with custom CUDA kernels
- **`transform_kernel.cu`** - Custom CUDA kernels for matrix transformations
- **`fpc.py`** - Main application with camera controls and rendering loop
- **`grid_renderer.py`** - 3D coordinate grid visualization for reference
- **`skybox_renderer.py`** - Skybox/background rendering system
- Other supporting files for complete 3D scene rendering

### What Has Been Achieved
1. **Custom CUDA Transform Pipeline**: Replaced NumPy/CuPy `@` operator with custom CUDA kernels
2. **Two-Stage Transformation**: Split MVP into separate MV and P matrix transformations
3. **GPU-Based Depth Sorting**: Points sorted by view-space z-coordinate (farthest to closest)
4. **PLY File Loading**: Basic point cloud loading with positions and spherical harmonic colors
5. **OpenGL Integration**: Vertex buffer management and shader compilation
6. **CuPy GPU Processing**: All transformations and sorting happen on GPU

### Current Rendering Method
- Points are rendered as GL_POINTS with fixed size
- Two-stage transformation: Model-View → sort by depth → Projection
- Colors come from spherical harmonic DC terms (0-order SH)
- Debug output shows transformation pipeline working correctly

## Next Steps: Implementing Gaussian Splatting

### Phase 1: Core Gaussian Data ✅ COMPLETED
- [x] **Load covariance data**: Extract scale vectors and rotation quaternions from PLY file
- [x] **Load opacity data**: Extract opacity values from PLY file  
- [x] **Data structure refactor**: Organize scale, rotation, opacity alongside positions/colors
- [x] **PLY structure analysis**: Added comprehensive debug output for field detection
- [x] **Robust field loading**: Safe field access with fallbacks for missing data
- [x] **Data preprocessing**: Proper quaternion normalization, scale handling, opacity sigmoid conversion

### Phase 2: 2D Covariance Projection
- [ ] **3D covariance construction**: Build 3D covariance matrices from scale/rotation (in its own kernel)
- [ ] **2D covariance computation**: Project 3D covariance to screen space using Jacobian (in kernel from above)
- [ ] **Eigenvalue decomposition**: Compute major/minor axes for quad sizing (in kernel from above)
- [ ] **Culling logic**: Skip Gaussians that are too small or behind camera

### Phase 3: Quad Generation
- [ ] **Quad vertex generation**: Create 4 vertices per Gaussian sized to 3σ
- [ ] **Instanced rendering setup**: Use OpenGL instancing for efficient quad rendering
- [ ] **UV coordinate mapping**: Pass quad-relative coordinates to fragment shader
- [ ] **Geometry buffer management**: Update VBO with quad vertices instead of points

### Phase 4: Fragment Shader Gaussian Evaluation
- [ ] **Update vertex shader**: Handle quad vertices and pass data to fragment shader
- [ ] **Fragment shader rewrite**: Compute Gaussian value `exp(-0.5 * d^T * Σ^(-1) * d)`
- [ ] **Alpha blending setup**: Enable proper transparency for overlapping Gaussians
- [ ] **Optimization**: Efficient inverse covariance computation in shader

### Phase 5: Rendering Pipeline Integration  
- [ ] **CUDA kernel updates**: Extend kernels to handle covariance transformations
- [ ] **Memory optimization**: Efficient GPU memory layout for all Gaussian attributes
- [ ] **Performance profiling**: Measure and optimize bottlenecks
- [ ] **Quality validation**: Compare results with reference Gaussian splatting

### Phase 6: Advanced Features (Future)
- [ ] **Higher-order SH**: Implement view-dependent color evaluation
- [ ] **Level-of-detail**: Adaptive quality based on distance/size
- [ ] **Frustum culling**: Cull Gaussians outside view frustum
- [ ] **Temporal coherence**: Optimize for animated/moving cameras

## Technical Notes

### Key Mathematical Components
1. **3D Covariance**: `Σ_3D = R * S * S^T * R^T` (rotation R, scale S)
2. **2D Projection**: `Σ_2D = J * W * Σ_3D * W^T * J^T` (Jacobian J, view matrix W)
3. **Gaussian Evaluation**: `α * exp(-0.5 * (p - μ)^T * Σ_2D^(-1) * (p - μ))`

### Rendering Strategy
- Continue using CuPy for GPU computations
- Maintain two-stage transformation pipeline
- Keep depth sorting for proper alpha blending
- Use fragment shaders for per-pixel Gaussian evaluation
- Target real-time performance with large point clouds

### Memory Layout Considerations
- Interleaved vs. separate arrays for attributes
- GPU memory bandwidth optimization
- Minimize CPU↔GPU transfers
- Efficient sorting of multiple attribute arrays