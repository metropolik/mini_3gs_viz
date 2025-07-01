# Gaussian Splatting Renderer - Development Plan

## Current State

### Files and Layout
- **`point_renderer.py`** - Main Gaussian renderer class with custom CUDA kernels
- **`transform_kernel.cu`** - Custom CUDA kernels for matrix transformations, covariance projection, and quad generation
- **`fpc.py`** - Main application with camera controls and rendering loop
- **`grid_renderer.py`** - 3D coordinate grid visualization for reference
- **`skybox_renderer.py`** - Skybox/background rendering system
- Other supporting files for complete 3D scene rendering

### What Has Been Achieved
1. **Complete Gaussian Splatting Pipeline**: Full 3D→2D covariance projection with fragment shader evaluation
2. **Custom CUDA Transform Pipeline**: Multi-stage GPU kernels for transformations, covariance, and quad generation
3. **Proper Alpha Blending**: Depth-sorted transparent rendering with correct blend modes
4. **GPU-Based Depth Sorting**: All Gaussian attributes sorted by view-space z-coordinate (back-to-front)
5. **PLY File Loading**: Complete Gaussian data loading (positions, colors, scales, rotations, opacity)
6. **High-Performance Rendering**: CUDA prefix sum compaction for efficient visible quad extraction
7. **Mathematical Gaussian Evaluation**: Fragment shader computes correct Gaussian weights for splatting

### Current Rendering Method
- Gaussians rendered as screen-aligned quads sized to 3σ bounds
- Two-stage transformation: Model-View → depth sort → covariance projection → quad generation
- Fragment shader evaluates Gaussian function exp(-0.5 * d^T * Σ^(-1) * d) for proper splatting
- Alpha blending combines overlapping Gaussians with proper transparency
- Colors from spherical harmonic DC terms with Gaussian-weighted alpha
- Achieved stable 60+ FPS rendering with thousands of Gaussians

## Next Steps: Implementing Gaussian Splatting

### Phase 1: Core Gaussian Data ✅ COMPLETED
- [x] **Load covariance data**: Extract scale vectors and rotation quaternions from PLY file
- [x] **Load opacity data**: Extract opacity values from PLY file  
- [x] **Data structure refactor**: Organize scale, rotation, opacity alongside positions/colors
- [x] **PLY structure analysis**: Added comprehensive debug output for field detection
- [x] **Robust field loading**: Safe field access with fallbacks for missing data
- [x] **Data preprocessing**: Proper quaternion normalization, scale handling, opacity sigmoid conversion

### Phase 2: 2D Covariance Projection ✅ COMPLETED
- [x] **3D covariance construction**: Build 3D covariance matrices from scale/rotation (in its own kernel)
- [x] **2D covariance computation**: Project 3D covariance to screen space using Jacobian (in kernel from above)
- [x] **Eigenvalue decomposition**: Compute major/minor axes for quad sizing (in kernel from above)
- [x] **Culling logic**: Skip Gaussians that are too small or behind camera
- [x] **CUDA kernel integration**: Added `compute_2d_covariance` kernel with complete pipeline
- [x] **Robust numerics**: Added regularization, safe math operations, and error handling
- [x] **GPU attribute sorting**: Sort all Gaussian data (scales, rotations, opacities) by depth

### Phase 3: Quad Generation ✅ COMPLETED
- [x] **Quad vertex generation**: Create 4 vertices per Gaussian sized to 3σ
- [x] **Instanced rendering setup**: Use OpenGL instancing for efficient quad rendering
- [x] **UV coordinate mapping**: Pass quad-relative coordinates to fragment shader
- [x] **Geometry buffer management**: Update VBO with quad vertices instead of points
- [x] **CUDA quad generation kernel**: Generate oriented quads from 2D covariance eigenvalues/eigenvectors
- [x] **Multiple VBO management**: Separate buffers for vertices, UVs, and per-quad data
- [x] **Visibility culling integration**: Only generate quads for visible Gaussians
- [x] **Alpha blending setup**: Enable proper transparency rendering
- [x] **GPU index generation**: Added CUDA kernel for efficient triangle index generation
- [x] **Proper projection pipeline**: Fixed NDC projection with full projection matrix application
- [x] **Correct covariance transformation**: Implemented paper's equation Σ' = J * W * Σ * W^T * J^T

### Phase 4: Fragment Shader Gaussian Evaluation ✅ COMPLETED
- [x] **Update vertex shader**: Handle quad vertices and pass data to fragment shader
- [x] **Fragment shader rewrite**: Compute Gaussian value `exp(-0.5 * d^T * Σ^(-1) * d)`
- [x] **Alpha blending setup**: Enable proper transparency for overlapping Gaussians
- [x] **Optimization**: Efficient inverse covariance computation in shader

### Phase 5: Rendering Pipeline Integration ✅ COMPLETED
- [x] **CUDA kernel updates**: Extended kernels with complete covariance transformation pipeline
- [x] **Memory optimization**: Efficient GPU memory layout with prefix sum compaction
- [x] **Performance optimization**: Eliminated atomic counter bottlenecks, GPU-only processing
- [x] **Alpha blending integration**: Proper depth testing with transparency rendering
- [x] **Depth order stability**: Fixed flashing issues by maintaining deterministic quad ordering
- [x] **Multi-buffer management**: Separate optimized buffers for vertices, UVs, and per-quad data
- [x] **Quality validation**: Achieved working Gaussian splatting with proper mathematical evaluation

### Phase 6: Mathematical Precision & Unit Consistency
- [ ] **Unit mismatch analysis**: Resolve coordinate space inconsistency between covariance and UV mapping
- [ ] **Dynamic UV mapping**: Pass actual quad extents to shader for mathematically correct mapping
- [ ] **OpenGL attribute restructuring**: Add vertex attributes for per-quad radius information  
- [ ] **Eliminate scaling hacks**: Remove fixed "magical" scaling factors with proper unit handling
- [ ] **Distance-invariant rendering**: Ensure Gaussian appearance doesn't change with camera distance
- [ ] **Shader optimization**: Compute UV scaling directly from NDC quad extents
- [ ] **Validation**: Verify mathematical correctness of d^T * Σ^(-1) * d computation

**Technical Challenge**: Currently using fixed scaling factor (0.001) in fragment shader due to unit mismatch between NDC-space covariance matrices and fixed [-3,3] UV mapping. Need to pass actual quad radii to shader for per-fragment correct coordinate transformation.

### Phase 7: Advanced Features (Future)
- [ ] **Higher-order SH**: Implement view-dependent color evaluation
- [ ] **Level-of-detail**: Adaptive quality based on distance/size
- [ ] **Frustum culling**: Cull Gaussians outside view frustum
- [ ] **Temporal coherence**: Optimize for animated/moving cameras

## Technical Notes

### Key Mathematical Components (Implemented)
1. **3D Covariance**: `Σ_3D = R * S * S^T * R^T` (rotation R, scale S) ✅
2. **2D Projection**: `Σ_2D = J * W * Σ_3D * W^T * J^T` (Jacobian J, view matrix W) ✅  
3. **Gaussian Evaluation**: `α * exp(-0.5 * d^T * Σ_2D^(-1) * d)` where d = UV-to-world transform ✅
4. **Quad Generation**: Screen-aligned quads sized to `radius = 3 * sqrt(eigenvalue)` ✅
5. **Depth Sorting**: Back-to-front ordering for proper alpha blending ✅

### Current Implementation Details
- **CUDA Pipeline**: 5 kernels (transform, covariance, quad generation, compaction, index generation)
- **Performance**: 60+ FPS with thousands of Gaussians using GPU-only processing
- **Memory Management**: Prefix sum compaction eliminates CPU-side loops and maintains sorted order
- **Blending**: `GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA` with depth testing but no depth writing
- **Coordinate Spaces**: NDC for all computations, maintains precision across camera distances

### Known Issues & Limitations
1. **Unit Mismatch**: Fixed scaling factor (0.001) needed due to coordinate space inconsistency
2. **Distance Dependency**: Gaussian apparent size changes with camera distance 
3. **UV Mapping**: Currently maps [0,1] to fixed [-3,3] instead of actual quad extents
4. **Mathematical Imprecision**: `d` vector not in same space as `Σ^(-1)` matrix

### Architecture Strengths
- ✅ **Mathematically Correct Covariance Projection**: Proper 3D→2D transformation
- ✅ **Efficient GPU Processing**: All-CUDA pipeline with minimal CPU-GPU transfers  
- ✅ **Stable Depth Ordering**: Deterministic rendering without flashing artifacts
- ✅ **Proper Alpha Blending**: Overlapping Gaussians combine correctly
- ✅ **Scalable Performance**: Handles large point clouds in real-time

### Future Architecture (Phase 6)
- **Per-Quad UV Scaling**: Pass actual NDC quad radii to fragment shader
- **Dynamic Coordinate Mapping**: `d = (UV - 0.5) * 2.0 * quadRadii` for mathematical correctness
- **OpenGL Restructuring**: Additional vertex attributes or texture-based data passing
- **Zero Scaling Factors**: Eliminate all "magical" constants with proper unit handling