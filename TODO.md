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

### Phase 6: Fragment Shader Refactoring - Screen Space Evaluation

#### Current State
The fragment shader currently relies on **interpolated per-vertex data**:
- UV coordinates are generated per-vertex (0,0 to 1,1) and interpolated across each quad
- The fragment shader uses these interpolated UVs to determine position within the quad: `vec2 d = (fragUV - 0.5) * 2.0 * quadRadii`
- All per-Gaussian data (color, opacity, inverse covariance) is duplicated for each vertex
- This creates unnecessary data duplication and relies on barycentric interpolation

#### Target State  
Refactor to use **screen-space evaluation directly**:
- Fragment shader receives the same data for all fragments in a quad (no interpolation)
- Use `gl_FragCoord.xy` to get the fragment's actual screen position
- Pass only the Gaussian's 2D screen-space center position
- Calculate offset directly: `vec2 d = gl_FragCoord.xy - gaussianCenter.xy`
- More elegant, direct computation without UV mapping indirection

#### Implementation Steps (Small Incremental Changes)

##### Step 1: Add Screen-Space Center to Fragment Shader ✅ COMPLETED
- [x] **Pass Gaussian center position**: Add gaussian center as a new vertex attribute (duplicated per vertex for now)
- [x] **Update CUDA kernel**: Modify `generate_quad_vertices` to include screen-space center position
- [x] **Update fragment shader**: Add new uniform/attribute for center position
- [x] **Test dual rendering**: Show both UV-based and gl_FragCoord-based calculations side-by-side
- [x] **Validation**: Confirm both methods produce identical results

##### Step 2: Convert Fragment Shader to gl_FragCoord ✅ COMPLETED
- [x] **Fragment shader switch**: Replace UV-based calculation with `gl_FragCoord.xy - gaussianCenter.xy`
- [x] **Handle coordinate systems**: Convert gl_FragCoord from pixel coordinates to same space as gaussian center
- [x] **Remove UV dependencies**: Comment out UV-based code but keep UV data flow for now
- [x] **Visual validation**: Ensure Gaussians still render correctly
- [x] **Debug output**: Temporarily visualize distance calculations to verify correctness

##### Step 3: Prepare for Instanced Rendering ✅ COMPLETED
- [x] **Create instance data structure**: Define struct for per-Gaussian data (center, color, opacity, covariance)
- [x] **Generate instance buffer data**: Modify CUDA to output instance data array alongside quad vertices
- [x] **Add instance VBO**: Create new VBO for instance data (not yet used for rendering)
- [x] **Test data generation**: Verify instance data is correctly populated
- [x] **Keep current rendering**: Don't switch to instanced rendering yet

##### Step 4: Implement Instanced Rendering ✅ COMPLETED
- [x] **Create base quad geometry**: Single quad with 4 vertices centered at origin
- [x] **Set up instanced attributes**: Configure VAO with per-instance data
- [x] **Update vertex shader**: Use instance data to position/scale quads
- [x] **Switch to glDrawElementsInstanced**: Replace current draw call
- [x] **A/B testing**: Add toggle to switch between old and new rendering methods

##### Step 5: Optimize and Clean Up 🚧 IN PROGRESS
- [x] **Remove old quad generation**: Delete unused quad vertex generation from CUDA kernel calls
- [x] **Remove per-vertex duplication**: Clean up redundant data in vertex buffers
- [x] **Optimize vertex shader**: Compute proper quad sizing and orientation from covariance in shader
- [x] **Remove A/B toggle**: Always use instanced rendering
- [ ] **Remove unused functions**: Delete old rendering functions and shaders
- [ ] **Clean up CUDA kernels**: Remove unused kernel functions from the .cu file
- [ ] **Performance validation**: Ensure rendering is faster than before

##### Step 6: Address Edge Cases
- [ ] **Viewport handling**: Ensure gl_FragCoord works correctly with viewport changes
- [ ] **Resolution independence**: Test with different window sizes
- [ ] **Coordinate precision**: Verify no precision issues at screen edges
- [ ] **Depth handling**: Confirm depth testing still works correctly
- [ ] **Final cleanup**: Remove all unused code paths and debug prints

**Technical Note**: This change will also naturally resolve the unit consistency issues from the original Phase 6, as we'll be working directly in screen space throughout.

### Phase 7: Performance Optimizations (Future)
- [ ] **GPU-based visibility culling**: Implement efficient CUDA prefix sum for compacting visible quads
- [ ] **Frustum culling**: Cull Gaussians outside view frustum  
- [ ] **Screen-space culling**: Skip quads that are too small to see
- [ ] **Z-culling**: Skip Gaussians behind the camera
- [ ] **Hierarchical culling**: Use spatial data structures for faster culling

### Phase 8: Advanced Features (Future)
- [ ] **Higher-order SH**: Implement view-dependent color evaluation
- [ ] **Level-of-detail**: Adaptive quality based on distance/size
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