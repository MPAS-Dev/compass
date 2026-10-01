# Thwaites01to60 Mesh

Variable-resolution global ocean mesh with 200m refinement at Thwaites grounding zone.

## Current Status

### ✅ Implemented:
1. **Base mesh generation** (`__init__.py`)
   - `Thwaites01to60BaseMesh` class
   - **Grounding line extraction from BedMachine** (automatic, no geojson needed)
   - Calculates flotation criterion: thickness > -bed × (ρ_ocean/ρ_ice)
   - Identifies grounding line as boundary between grounded/floating ice
   - Creates signed distance field for mesh refinement
   - Falls back to Gaussian approximation if BedMachine unavailable
   - Configurable resolutions: 200m GZ, 1km cavity, 3km shelf, 8km far-field

2. **Configuration** (`thwaites01to60.cfg`)
   - Resolution parameters
   - Regional domain settings (enabled by default for Amundsen sector)
   - Thin film parameters (disabled by default)

3. **Regional culling** ✅ **FULLY INTEGRATED**
   - `ThwaitesCullMeshStep` custom cull step (`cull_mesh.py`)
   - Hook-based integration into standard culling workflow
   - Modifies land mask to mark cells outside domain as land
   - Supports geojson polygon or lat-lon bounding box
   - **Working and ready to test**

4. **Registration**
   - Mesh registered in `compass/ocean/tests/global_ocean/mesh/__init__.py`
   - Custom CullMeshStep wired in for Thwaites meshes
   - Ready for `compass list` and `compass setup`

### 🚧 TODO:

#### High Priority:
1. ~~**Integrate regional culling into CullMeshStep**~~ ✅ **DONE**
   - ✅ Added hook point in base `cull.py`
   - ✅ Created `ThwaitesCullMeshStep` with hook file generation
   - ✅ Wired into Mesh test case for Thwaites meshes
   - ✅ Ready for testing

2. ~~**Replace Gaussian refinement with grounding-line-based refinement**~~ ✅ **DONE**
   - ✅ Extracts grounding line directly from BedMachine
   - ✅ Calculates flotation criterion automatically
   - ✅ Uses `signed_distance_from_geojson` for accurate GZ band
   - ✅ No manual geojson file required
   - ⚠️ Requires BedMachine file in bathymetry database

#### Medium Priority:
3. **Add thin film support**
   - Modify land mask to keep cells beneath grounded ice
   - Based on height above flotation from BedMachine
   - Requires integration with `remap_topography` step

4. **Testing**
   - Set up and run basic mesh generation
   - Validate cell widths and resolution
   - Test regional culling with lat-lon bounds
   - Test with/without thin film

#### Low Priority:
5. **Clean up copied FRIS files**
   - Remove unused geojson files (atlantic.geojson, fris_*.geojson, etc.)
   - Remove or adapt FRIS-specific namelists if needed

## Usage

### Basic mesh (no regional culling):
```bash
compass list | grep Thwaites
compass setup -t global_ocean/mesh/Thwaites01to60 -w $WORK
compass run $WORK
```

### With regional domain (enabled by default):
Regional culling is **enabled by default** for Amundsen sector (76-73°S, 116-98°W).

To change the domain, edit `thwaites01to60.cfg`:
```ini
[thwaites01to60]
# Option A: lat-lon bounds (currently enabled)
lat_min = -76.0
lat_max = -73.0
lon_min = -116.0
lon_max = -98.0

# Option B: geojson polygon
# regional_domain_geojson = amundsen_domain.geojson
```

To disable regional culling (full global mesh), comment out all domain options.

## Implementation Notes

### Mesh Refinement:
The implementation now extracts the grounding line directly from BedMachine:

1. **Load BedMachine data** (ice thickness, bed elevation)
2. **Calculate flotation criterion**:
   ```python
   # Ice is grounded when: thickness > -bed × (ρ_ocean/ρ_ice)
   thickness_flotation = -bed * (1028.0 / 918.0)  # where bed < 0
   grounded = thickness > thickness_flotation
   ```
3. **Identify grounding line**: Boundary between grounded and floating ice
4. **Create feature collection**: Convert GL cells to geojson polygon
5. **Calculate signed distance**: Use `signed_distance_from_geojson`
6. **Apply refinement**:
   - GZ band: ±15 km from GL → 200m resolution
   - Cavity: 100 km into cavity → 1 km resolution  
   - Shelf: 200 km into cavity → 3 km resolution
   - Far-field: everywhere else → 8 km resolution

**Fallback**: If BedMachine is unavailable, uses Gaussian approximation around Thwaites location (75°S, 106°W).

### Regional Culling:
**Fully implemented and integrated!**

The `ThwaitesCullMeshStep` (in `cull_mesh.py`) creates a hook file that:
1. Is executed by the modified `cull.py` after land mask creation
2. Loads base mesh and land mask
3. Determines cells inside/outside regional domain
4. Marks outside cells as "land"
5. Standard culling then removes these cells

The hook runs between lines 307 and 308 of `compass/ocean/mesh/cull.py`,
after `land_mask.nc` is created but before culling begins.

### Thin Film:
Will require:
1. BedMachine topography (from `remap_topography` step)
2. Compute height above flotation for grounded ice
3. Cells with HAF < threshold (e.g., 30m) kept as ocean
4. Add `thinFilmMask` field for diagnostics

## References

- Design document: `/Users/trhille/Documents/Antarctica/thwaites_grounding_zone_intrusion/Grounding_zone_intrusion_simulation_design.pdf`
- Implementation plan: `/Users/trhille/Documents/Antarctica/thwaites_grounding_zone_intrusion/QUICK_START.md`
- FRIS mesh (template): `compass/ocean/tests/global_ocean/mesh/fris01to60/`
