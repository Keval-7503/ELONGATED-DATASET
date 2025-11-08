"""
render_glb_dataset.py

Complete Blender script to render RGB, mask, and depth without any background
compositing. Outputs are organized into train/val/test, each with rgb/mask/depth
subfolders. Depth is saved as OpenEXR.

Run from Blender like:
  "C:\\Users\\KEVAL\\Downloads\\blender-4.5.4-candidate+v45.57e187f8f5cd-windows.amd64-release\\blender-4.5.4-candidate+v45.57e187f8f5cd-windows.amd64-release\\blender.exe" \
    -b -P render_glb_dataset.py -- \
    --model "D:\\CV_DATASET\\model.glb" \
    --out_root "D:\\CV_DATASET\\dataset" \
    --n 500 --res 512 \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
    --engine cycles --seed 42
"""

import os
import sys
import math
import argparse
import random
from typing import List, Tuple, Optional


def _argv_after_double_dash() -> list:
    """Return arguments after the `--` Blender delimiter.

    Blender consumes its own CLI flags before `--`. Everything after `--`
    is forwarded to this script. If `--` is missing, return an empty list.
    """
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--") + 1
        return argv[idx:]
    return []


def eprint(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def wprint(msg: str) -> None:
    print(f"[WARN] {msg}")


def _try_import_bpy():
    try:
        import bpy  # type: ignore
        return bpy
    except Exception:
        return None


def ensure_clean_scene(bpy) -> None:
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Purge orphan data to keep things tidy
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def import_glb_gltf(bpy, model_path: str) -> List[object]:
    before = set(o.name for o in bpy.data.objects)
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".glb", ".gltf"):
        res = bpy.ops.import_scene.gltf(filepath=model_path)
        if res != {'FINISHED'}:
            wprint(f"GLTF import returned: {res}")
    else:
        eprint("Unsupported model extension; expected .glb/.gltf")
        return []
    after_objs = [o for o in bpy.data.objects if o.name not in before]
    return after_objs


def make_model_collection(bpy, objects: List[object], collection_name: str = "MODEL") -> None:
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)
    # Link objects and unlink from other collections to keep them grouped
    for obj in objects:
        if obj.users_collection:
            for c in list(obj.users_collection):
                try:
                    c.objects.unlink(obj)
                except Exception:
                    pass
        if obj.name not in coll.objects:
            coll.objects.link(obj)


def set_pass_index(bpy, objects: List[object], index: int = 1) -> None:
    for obj in objects:
        try:
            obj.pass_index = index
        except Exception:
            pass


def _select_only(bpy, objects: List[object]) -> None:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        try:
            obj.select_set(True)
        except Exception:
            pass
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def apply_rot_scale(bpy, objects: List[object]) -> None:
    _select_only(bpy, objects)
    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    except Exception as ex:
        wprint(f"Failed to apply transforms: {ex}")


def _world_bbox_points(bpy, obj) -> List[Tuple[float, float, float]]:
    # obj.bound_box is 8 corner points in local space; transform to world
    pts = []
    for co in obj.bound_box:
        v = obj.matrix_world @ bpy.mathutils.Vector(co)
        pts.append((v.x, v.y, v.z))
    return pts


def all_world_bbox_points(bpy, objects: List[object]) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    for obj in objects:
        try:
            pts.extend(_world_bbox_points(bpy, obj))
        except Exception:
            pass
    return pts


def _combine_bounds(points: List[Tuple[float, float, float]]):
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    min_z = min(p[2] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    max_z = max(p[2] for p in points)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def center_to_origin_world(bpy, objects: List[object]) -> None:
    all_pts: List[Tuple[float, float, float]] = []
    for obj in objects:
        try:
            all_pts.extend(_world_bbox_points(bpy, obj))
        except Exception:
            pass
    if not all_pts:
        return
    (min_x, min_y, min_z), (max_x, max_y, max_z) = _combine_bounds(all_pts)
    cx, cy, cz = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z))
    # Translate each object so that bbox center moves to world origin
    for obj in objects:
        try:
            obj.location.x -= cx
            obj.location.y -= cy
            obj.location.z -= cz
        except Exception:
            pass


def set_origin_to_geometry(bpy, objects: List[object]) -> None:
    _select_only(bpy, objects)
    try:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    except Exception as ex:
        wprint(f"Failed to set origin to geometry: {ex}")


def scale_to_unit_max_dim(bpy, objects: List[object], target_max: float = 1.0) -> None:
    # Compute combined bbox in world
    all_pts: List[Tuple[float, float, float]] = []
    for obj in objects:
        try:
            all_pts.extend(_world_bbox_points(bpy, obj))
        except Exception:
            pass
    if not all_pts:
        return
    (min_x, min_y, min_z), (max_x, max_y, max_z) = _combine_bounds(all_pts)
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z
    max_dim = max(dim_x, dim_y, dim_z, 1e-9)
    scale = target_max / max_dim
    for obj in objects:
        try:
            obj.scale.x *= scale
            obj.scale.y *= scale
            obj.scale.z *= scale
        except Exception:
            pass
    # Apply scale
    apply_rot_scale(bpy, objects)


def compute_bounds_info(bpy, objects: List[object]) -> Tuple[float, Tuple[float, float, float]]:
    all_pts: List[Tuple[float, float, float]] = []
    for obj in objects:
        try:
            all_pts.extend(_world_bbox_points(bpy, obj))
        except Exception:
            pass
    if not all_pts:
        return 0.5, (1.0, 1.0, 1.0)
    (min_x, min_y, min_z), (max_x, max_y, max_z) = _combine_bounds(all_pts)
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z
    # Approximate bounding-sphere radius as half the diagonal of the bbox
    diag = math.sqrt(dim_x * dim_x + dim_y * dim_y + dim_z * dim_z)
    radius = max(1e-9, 0.5 * diag)
    return radius, (dim_x, dim_y, dim_z)


def fibonacci_sphere_directions(n: int, upper_hemisphere: bool = True) -> List[Tuple[float, float, float]]:
    """Generate n approximately uniform directions on a unit sphere.

    If upper_hemisphere is True, constrain directions to z >= 0 (keeps camera above).
    """
    if n <= 0:
        return []
    dirs: List[Tuple[float, float, float]] = []
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(n):
        z = 1.0 - (2.0 * (i + 0.5) / n)
        if upper_hemisphere:
            z = abs(z)  # reflect to upper hemisphere
        r = max(0.0, 1.0 - z * z)
        r = math.sqrt(r)
        theta = ga * i
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        dirs.append((x, y, z))
    return dirs


def build_split_sequence(train_count: int, val_count: int, test_count: int) -> List[str]:
    """Deterministic order: all train, then val, then test."""
    return (['train'] * int(train_count)) + (['val'] * int(val_count)) + (['test'] * int(test_count))


def generate_camera_params(n: int, bs_radius: float, rng: random.Random, cam_obj=None, base_override: Optional[float]=None):
    """Precompute per-sample camera distance (fit-based) and roll=0.

    Distances are chosen so the object occupies different scales while staying centered:
      - close:   scale ~ 0.85..0.95 (larger in frame)
      - medium:  scale ~ 0.65..0.80
      - far:     scale ~ 0.45..0.60 (smaller in frame)

    distance = (radius / tan(fov/2)) / scale
    Uses vertical FOV when available to ensure full fit.
    """
    # Determine camera vertical FOV
    fov = math.radians(40.0)
    try:
        if cam_obj is not None and hasattr(cam_obj.data, 'angle_y'):
            fov = float(cam_obj.data.angle_y)
        elif cam_obj is not None and hasattr(cam_obj.data, 'angle'):
            fov = float(cam_obj.data.angle)
    except Exception:
        pass
    base = bs_radius / max(1e-6, math.tan(0.5 * fov))
    if base_override is not None and base_override > 0:
        base = float(base_override)

    # Build target scales list (close/medium/far) evenly distributed
    scales: List[float] = []
    thirds = max(1, n // 3)
    for _ in range(thirds):
        scales.append(rng.uniform(0.97, 0.995))  # close (very big)
    for _ in range(thirds):
        scales.append(rng.uniform(0.90, 0.95))   # medium (big)
    while len(scales) < n:
        scales.append(rng.uniform(0.82, 0.88))   # far (still fairly big)
    # Shuffle for variety
    rng.shuffle(scales)
    distances = [base / s for s in scales[:n]]
    rolls = [0.0 for _ in range(n)]  # keep level to keep object centered upright
    return distances, rolls


def setup_render_engine(bpy, engine: str, res: int) -> None:
    scn = bpy.context.scene
    scn.render.resolution_x = int(res)
    scn.render.resolution_y = int(res)
    scn.render.resolution_percentage = 100
    # Avoid stray primary renders in split folders
    try:
        import tempfile
        tmp_dir = tempfile.gettempdir()
        scn.render.filepath = os.path.join(tmp_dir, "blender_main_")
        scn.render.use_file_extension = True
    except Exception:
        pass
    # Use visible white world, not transparent film
    try:
        scn.render.film_transparent = False
    except Exception:
        pass
    # Color management for sRGB outputs
    try:
        scn.display_settings.display_device = 'sRGB'
        scn.view_settings.view_transform = 'Standard'
        scn.view_settings.look = 'None'
        scn.view_settings.exposure = 0.0
        scn.view_settings.gamma = 1.0
    except Exception:
        pass

    if engine == "cycles":
        scn.render.engine = 'CYCLES'
        try:
            scn.cycles.samples = 32
        except Exception:
            pass
        # Ensure shadows are enabled and soft-caustics off for stability
        try:
            scn.cycles.use_motion_blur = False
        except Exception:
            pass
        # View layer denoising if available
        try:
            bpy.context.view_layer.cycles.use_denoising = True
        except Exception:
            try:
                scn.cycles.use_denoising = True  # legacy fallback
            except Exception:
                pass
        # Attempt GPU enable; fall back to CPU silently
        try:
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons['cycles'].preferences
            # Prefer OPTIX, then CUDA, then HIP/ONEAPI if present
            for backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):  # order of preference
                try:
                    cycles_prefs.compute_device_type = backend
                    break
                except Exception:
                    continue
            # Enable all available devices
            for dev in cycles_prefs.get_devices():
                for d in dev:
                    d.use = True
            try:
                scn.cycles.device = 'GPU'
            except Exception:
                pass
        except Exception:
            # Leave at CPU
            pass
    else:
        scn.render.engine = 'BLENDER_EEVEE'
        try:
            scn.eevee.taa_render_samples = 32
            scn.eevee.use_gtao = True
            scn.eevee.use_bloom = False
            scn.eevee.shadow_cube_size = '1024'
            scn.eevee.shadow_cascade_size = '1024'
        except Exception:
            pass


def create_camera_with_target(bpy, focal_length_mm: float = 50.0):
    cam_data = bpy.data.cameras.new(name='CAMERA')
    cam_obj = bpy.data.objects.new('CAMERA', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_data.lens = float(focal_length_mm)
    # Sensible clipping for our scene scale
    try:
        cam_data.clip_start = 0.01
        cam_data.clip_end = 100.0
        cam_data.shift_x = 0.0
        cam_data.shift_y = 0.0
        cam_data.sensor_fit = 'AUTO'
    except Exception:
        pass
    # Create target empty at origin
    tgt = bpy.data.objects.new('CAM_TARGET', None)
    tgt.empty_display_size = 0.2
    tgt.empty_display_type = 'PLAIN_AXES'
    bpy.context.scene.collection.objects.link(tgt)
    # Position camera to a reasonable default
    cam_obj.location = (0.0, -2.5, 1.25)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    # Add Track To constraint to look at origin
    tr = cam_obj.constraints.new(type='TRACK_TO')
    tr.target = tgt
    tr.track_axis = 'TRACK_NEGATIVE_Z'
    tr.up_axis = 'UP_Y'
    # Make this camera active for rendering
    try:
        bpy.context.scene.camera = cam_obj
    except Exception:
        pass
    return cam_obj, tgt


def get_model_center(bpy, objects: List[object]) -> Tuple[float, float, float]:
    all_pts: List[Tuple[float, float, float]] = []
    for obj in objects:
        try:
            all_pts.extend(_world_bbox_points(bpy, obj))
        except Exception:
            pass
    if not all_pts:
        return (0.0, 0.0, 0.0)
    (min_x, min_y, min_z), (max_x, max_y, max_z) = _combine_bounds(all_pts)
    cx, cy, cz = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z))
    return (cx, cy, cz)


def is_bbox_in_view(bpy, scene, cam_obj, bbox_points: List[Tuple[float, float, float]], margin: float = 0.02) -> bool:
    """Checks whether at least part of the bbox projects inside the camera frame.

    Uses bpy_extras.object_utils.world_to_camera_view if available.
    """
    try:
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
    except Exception:
        return True  # cannot assess; assume ok when not in Blender
    if not bbox_points:
        return True
    inside = 0
    for p in bbox_points:
        try:
            v = world_to_camera_view(scene, cam_obj, Vector(p))
            if (v.z > 0.0 and (margin <= v.x <= 1.0 - margin) and (margin <= v.y <= 1.0 - margin)):
                inside += 1
        except Exception:
            continue
    return inside > 0


def bbox_fill_fraction(bpy, scene, cam_obj, bbox_points: List[Tuple[float, float, float]]) -> float:
    """Return approximate screen fill (max of width or height) of the bbox in [0, 1+].

    If nothing projects in front of the camera, return a large number (>1) to signal too close/behind.
    """
    try:
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
    except Exception:
        return 1.0
    if not bbox_points:
        return 1.0
    xs, ys = [], []
    any_front = False
    for p in bbox_points:
        try:
            v = world_to_camera_view(scene, cam_obj, Vector(p))
            if v.z > 0.0:
                xs.append(v.x)
                ys.append(v.y)
                any_front = True
        except Exception:
            continue
    if not any_front:
        return 2.0  # behind camera or degenerate
    w = max(0.0, min(1.0, max(xs, default=0.0)) - max(0.0, min(xs, default=1.0)))
    h = max(0.0, min(1.0, max(ys, default=0.0)) - max(0.0, min(ys, default=1.0)))
    return max(w, h)


def adjust_distance_to_fill(bpy, scene, cam_obj, dir_vec, start_dist: float,
                            bbox_points: List[Tuple[float, float, float]],
                            target_fill: float,
                            fit_distance: float,
                            bs_radius: float,
                            min_fit_frac: float,
                            iters: int = 4) -> float:
    """Iteratively adjust camera distance along its viewing ray to hit target fill.

    Keeps constraints: dist >= max(1.05*radius, min_fit_frac*fit_distance)
                       dist <= 1.25*fit_distance (avoid going too far)
    """
    dist = float(start_dist)
    floor_dist = max(1.05 * bs_radius, float(min_fit_frac) * float(fit_distance))
    ceil_dist = 1.25 * float(fit_distance)
    for _ in range(max(1, iters)):
        cam_obj.location = (dir_vec[0] * dist, dir_vec[1] * dist, dir_vec[2] * dist)
        fill = bbox_fill_fraction(bpy, scene, cam_obj, bbox_points)
        if fill <= 0.0:
            break
        # If behind camera or crazy, push outwards
        if fill > 1.2:
            dist *= 1.10
        else:
            # distance approximately inversely proportional to fill
            scale = fill / max(1e-4, target_fill)
            # damped adjustment
            dist = dist * (0.5 + 0.5 * scale)
        # Enforce bounds
        dist = max(dist, floor_dist)
        dist = min(dist, ceil_dist)
    return dist


def get_camera_fov_xy(bpy, cam_obj, scene) -> Tuple[float, float]:
    cam = cam_obj.data
    # Prefer Blender-provided angles if available
    try:
        ax = float(getattr(cam, 'angle_x'))
        ay = float(getattr(cam, 'angle_y'))
        if ax > 0 and ay > 0:
            return ax, ay
    except Exception:
        pass
    # Compute from lens and sensor
    try:
        import math as _m
        lens = float(cam.lens)  # mm
        sw = float(cam.sensor_width)  # mm
        sh = float(cam.sensor_height)  # mm
        ax = 2.0 * _m.atan((sw * 0.5) / lens)
        ay = 2.0 * _m.atan((sh * 0.5) / lens)
        return ax, ay
    except Exception:
        return (math.radians(49.0), math.radians(36.0))


def compute_fit_distance_for_radius(radius: float, fov_x: float, fov_y: float, margin: float = 1.2) -> float:
    # Ensure a sphere of given radius fits within both horizontal and vertical FOV
    tx = math.tan(max(1e-6, fov_x * 0.5))
    ty = math.tan(max(1e-6, fov_y * 0.5))
    min_tan = max(1e-6, min(tx, ty))
    base = radius / min_tan
    return float(margin) * base


def compute_fit_distance_for_bbox(dim_x: float, dim_y: float, fov_x: float, fov_y: float, margin: float = 1.0) -> float:
    """Distance so that a bounding box fits within the view frustum.

    Uses horizontal and vertical half extents and FOVs.
    """
    tx = math.tan(max(1e-6, fov_x * 0.5))
    ty = math.tan(max(1e-6, fov_y * 0.5))
    need_x = (0.5 * dim_x) / max(1e-6, tx)
    need_y = (0.5 * dim_y) / max(1e-6, ty)
    base = max(need_x, need_y)
    return float(margin) * base


def create_sun_light(bpy, strength: float = 3.0):
    light_data = bpy.data.lights.new(name='SUN', type='SUN')
    light_data.energy = float(strength)
    try:
        light_data.angle = math.radians(1.0)  # crisper shadows
    except Exception:
        pass
    light_obj = bpy.data.objects.new('SUN', light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    # Default angle
    light_obj.rotation_euler = (math.radians(55.0), math.radians(0.0), math.radians(35.0))
    light_obj.location = (2.0, -2.0, 3.0)
    # Eevee: ensure contact shadows enabled for tighter contact
    try:
        light_data.use_contact_shadow = True
        light_data.contact_shadow_distance = 0.2
        light_data.contact_shadow_bias = 0.03
    except Exception:
        pass
    return light_obj


def add_shadow_catcher_plane(bpy, size: float = 4.0, z_offset: float = -0.001):
    # Create a plane slightly below origin for shadows
    try:
        bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD',
                                         location=(0.0, 0.0, z_offset))
        plane = bpy.context.active_object
        plane.name = 'SHADOW_CATCHER'
        # Mark as shadow catcher for Cycles
        try:
            plane.cycles.is_shadow_catcher = True
        except Exception:
            pass
        return plane
    except Exception as ex:
        wprint(f"Failed to add shadow catcher plane: {ex}")
        return None


def add_white_ground_plane(bpy, z: float, size: float = 6.0):
    try:
        bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD',
                                         location=(0.0, 0.0, z))
        plane = bpy.context.active_object
        plane.name = 'GROUND_WHITE'
        # Make it a Cycles shadow catcher so only shadows are visible (no wedge in RGB)
        try:
            plane.cycles.is_shadow_catcher = True
        except Exception:
            pass
        # Assign simple white material
        mat = bpy.data.materials.new(name='MAT_WHITE')
        try:
            mat.use_nodes = True
            nt = mat.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
            # Slightly off-white so shadows/edges are visible
            bsdf.inputs['Base Color'].default_value = (0.92, 0.92, 0.92, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.55
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        except Exception:
            pass
        if plane.data.materials:
            plane.data.materials[0] = mat
        else:
            plane.data.materials.append(mat)
        return plane
    except Exception as ex:
        wprint(f"Failed to add white ground plane: {ex}")
        return None


def ensure_viewlayer_passes(bpy) -> None:
    vl = bpy.context.view_layer
    try:
        vl.use_pass_object_index = True
    except Exception:
        pass
    try:
        vl.use_pass_z = True
    except Exception:
        pass


def build_compositor_tree(bpy):
    scn = bpy.context.scene
    scn.use_nodes = True
    nt = scn.node_tree
    # Clear existing nodes
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    # Nodes
    n_rl = nt.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-800, 200)

    n_id = nt.nodes.new('CompositorNodeIDMask')
    n_id.index = 1
    try:
        n_id.use_antialiasing = False
    except Exception:
        pass
    n_id.location = (-600, -100)

    # Separate File Output nodes for robust per-format control
    n_out_rgb = nt.nodes.new('CompositorNodeOutputFile')
    n_out_rgb.location = (200, 200)
    n_out_rgb.base_path = ""
    try:
        n_out_rgb.format.file_format = 'PNG'
        n_out_rgb.format.color_mode = 'RGB'
        n_out_rgb.format.color_depth = '8'
    except Exception:
        pass
    if n_out_rgb.file_slots:
        n_out_rgb.file_slots[0].path = 'img_'

    n_out_mask = nt.nodes.new('CompositorNodeOutputFile')
    n_out_mask.location = (200, -50)
    n_out_mask.base_path = ""
    try:
        n_out_mask.format.file_format = 'PNG'
        n_out_mask.format.color_mode = 'BW'
        n_out_mask.format.color_depth = '8'
    except Exception:
        pass
    if n_out_mask.file_slots:
        n_out_mask.file_slots[0].path = 'img_'

    n_out_depth = nt.nodes.new('CompositorNodeOutputFile')
    n_out_depth.location = (200, -300)
    n_out_depth.base_path = ""
    try:
        n_out_depth.format.file_format = 'OPEN_EXR'
        n_out_depth.format.color_mode = 'BW'
        n_out_depth.format.color_depth = '32'
        n_out_depth.format.exr_codec = 'ZIP'
    except Exception:
        pass
    if n_out_depth.file_slots:
        n_out_depth.file_slots[0].path = 'img_'

    # Links
    ln = nt.links
    # RGB from Render Layers “Image/Combined”
    try:
        rl_img_out = n_rl.outputs.get('Image', None) or n_rl.outputs.get('Combined', None) or n_rl.outputs[0]
    except Exception:
        rl_img_out = None
    if rl_img_out is not None:
        ln.new(rl_img_out, n_out_rgb.inputs[0])

    # Mask from ID Mask alpha
    ln.new(n_rl.outputs['IndexOB'], n_id.inputs['ID value'])
    ln.new(n_id.outputs['Alpha'], n_out_mask.inputs[0])

    # Depth
    ln.new(n_rl.outputs['Depth'], n_out_depth.inputs[0])

    return {
        'tree': nt,
        'nodes': {
            'rl': n_rl,
            'idmask': n_id,
            'out_rgb': n_out_rgb,
            'out_mask': n_out_mask,
            'out_depth': n_out_depth,
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Render GLB/GLTF dataset in Blender",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to .glb/.gltf model")
    parser.add_argument("--out_root", required=True, help="Dataset root output directory")

    parser.add_argument("--n", type=int, default=6, help="Total number of samples")
    parser.add_argument("--res", type=int, default=512, help="Square render resolution (px)")

    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")

    parser.add_argument("--engine", choices=["cycles", "eevee"], default="cycles",
                        help="Render engine")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--category", type=str, default="chain",
                        help="Optional subfolder under out_root to place splits")
    parser.add_argument("--zoom", type=float, default=0.80,
                        help="Fraction of tight-fit distance (smaller = closer/bigger object). Typical 0.30–1.00")
    parser.add_argument("--min_fit", type=float, default=0.80,
                        help="Minimum fraction of tight-fit distance allowed (avoids clipping). Lower => larger object")
    parser.add_argument("--fill", type=float, default=None,
                        help="Target on-screen fill for the object bounding box [0.50–0.98]. Overrides auto 0.88 if set.")

    args = parser.parse_args(_argv_after_double_dash())

    # Normalize paths early
    args.model = os.path.abspath(args.model)
    args.out_root = os.path.abspath(args.out_root)
    
    return args


def validate_args(args) -> bool:
    ok = True

    # Existence checks
    if not os.path.isfile(args.model):
        eprint(f"Model file not found: {args.model}")
        ok = False
    else:
        _, ext = os.path.splitext(args.model)
        if ext.lower() not in (".glb", ".gltf"):
            eprint("--model must be a .glb or .gltf file")
            ok = False


    # Ratios
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(r < 0.0 or r > 1.0 for r in ratios):
        eprint("Ratios must be within [0, 1]")
        ok = False
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        eprint(f"Ratios must sum to 1.0 (got {total:.8f})")
        ok = False

    # Counts and params
    if args.n is None or args.n < 3:
        eprint("--n must be an integer >= 3")
        ok = False

    if args.res is None or args.res < 1:
        eprint("--res must be a positive integer")
        ok = False

    if args.engine not in ("cycles", "eevee"):
        eprint("--engine must be 'cycles' or 'eevee'")
        ok = False

    try:
        if args.zoom <= 0.05 or args.zoom > 2.0:
            wprint("--zoom out of typical range (0.05, 2.0]; proceeding anyway")
        if args.min_fit <= 0.3 or args.min_fit > 1.5:
            wprint("--min_fit outside typical range (0.3, 1.5]; proceeding anyway")
        if args.fill is not None and not (0.4 <= args.fill <= 0.99):
            wprint("--fill recommended in [0.50, 0.98]; proceeding anyway")
    except Exception:
        pass

    return ok


def compute_split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    """Compute split counts deterministically so they sum to n.

    Strategy: floor counts and distribute the remainder by descending fractional parts.
    """
    targets = {
        "train": n * train_ratio,
        "val": n * val_ratio,
        "test": n * test_ratio,
    }
    floors = {k: int(math.floor(v)) for k, v in targets.items()}
    total_floor = sum(floors.values())
    remainder = n - total_floor
    # Order by fractional part (descending), then fixed tiebreaker order train, val, test
    frac = [(k, targets[k] - floors[k]) for k in ("train", "val", "test")]
    frac.sort(key=lambda kv: (-kv[1], (0 if kv[0]=="train" else 1 if kv[0]=="val" else 2)))

    counts = floors.copy()
    for i in range(remainder):
        counts[frac[i % len(frac)][0]] += 1
    return counts["train"], counts["val"], counts["test"]


def ensure_output_dirs(out_root: str):
    """Create dataset directory tree with rgb/mask/depth inside each split.

    Returns mapping: {split: {rgb, mask, depth}}
    """
    mapping = {}
    for split in ("train", "val", "test"):
        split_root = os.path.join(out_root, split)
        paths = {
            "rgb": os.path.join(split_root, "rgb"),
            "mask": os.path.join(split_root, "mask"),
            "depth": os.path.join(split_root, "depth"),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        mapping[split] = paths
    return mapping


# Backgrounds are intentionally not used. No listing function needed.


def main():
    args = parse_args()

    if not validate_args(args):
        sys.exit(1)

    random.seed(args.seed)

    # Compute split counts
    train_count, val_count, test_count = compute_split_counts(
        args.n, args.train_ratio, args.val_ratio, args.test_ratio
    )

    # Create directories under optional category subfolder
    out_root_eff = os.path.join(args.out_root, args.category) if getattr(args, 'category', None) else args.out_root
    dir_map = ensure_output_dirs(out_root_eff)

    # Report summary
    print("=== Dataset Setup ===")
    print(f"Model: {args.model}")
    print(f"Output root: {args.out_root}")
    if getattr(args, 'category', None):
        print(f"Category: {args.category}")
    fill_info = f" | Fill: {args.fill}" if getattr(args, 'fill', None) is not None else ""
    print(f"Engine: {args.engine} | Seed: {args.seed} | Zoom: {args.zoom} | MinFit: {args.min_fit}{fill_info}")
    print(f"Resolution: {args.res}x{args.res}")
    print(f"Total samples: {args.n}")
    print(f"Split counts -> train: {train_count}, val: {val_count}, test: {test_count}")
    print("Created directories:")
    for split in ("train", "val", "test"):
        print(f"  {split}/rgb: {dir_map[split]['rgb']}")
        print(f"  {split}/mask: {dir_map[split]['mask']}")
        print(f"  {split}/depth: {dir_map[split]['depth']}")

    # Attempt Blender-specific setup if running inside Blender.
    bpy = _try_import_bpy()
    if bpy is None:
        print("Blender (bpy) not available; skipping import/normalize in this step.")
        print("Next: run via Blender to proceed with scene setup.")
        return

    # Inside Blender: clean scene, import model, normalize, and report bounding info.
    ensure_clean_scene(bpy)
    imported = import_glb_gltf(bpy, args.model)
    if not imported:
        eprint("Failed to import any objects from the model.")
        sys.exit(1)
    make_model_collection(bpy, imported, collection_name="MODEL")
    set_pass_index(bpy, imported, index=1)
    apply_rot_scale(bpy, imported)
    set_origin_to_geometry(bpy, imported)
    center_to_origin_world(bpy, imported)
    scale_to_unit_max_dim(bpy, imported, target_max=1.0)
    set_origin_to_geometry(bpy, imported)
    center_to_origin_world(bpy, imported)
    bs_radius, bbox_dims = compute_bounds_info(bpy, imported)

    # Configure engine and basic scene features
    setup_render_engine(bpy, args.engine, args.res)
    ensure_viewlayer_passes(bpy)
    cam_obj, cam_tgt = create_camera_with_target(bpy, focal_length_mm=50.0)
    sun_obj = create_sun_light(bpy, strength=5.0)
    # Add a white ground plane at model base so shadows are visible
    # Determine base Z from bounds
    try:
        all_pts: List[Tuple[float, float, float]] = []
        for obj in imported:
            try:
                all_pts.extend(_world_bbox_points(bpy, obj))
            except Exception:
                pass
        min_z = min(p[2] for p in all_pts) if all_pts else -0.5
    except Exception:
        min_z = -0.5
    ground = add_white_ground_plane(bpy, z=min_z - 0.002, size=6.0)
    shadow_plane = ground
    comp_refs = build_compositor_tree(bpy)
    # Light the scene with a neutral world so frames never go black
    try:
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        world = bpy.context.scene.world
        world.use_nodes = True
        wnt = world.node_tree
        for n in list(wnt.nodes):
            wnt.nodes.remove(n)
        n_bg = wnt.nodes.new('ShaderNodeBackground')
        n_bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # white background
        n_bg.inputs[1].default_value = 0.05  # dimmer world so object and shadows are visible
        n_out = wnt.nodes.new('ShaderNodeOutputWorld')
        wnt.links.new(n_bg.outputs['Background'], n_out.inputs['Surface'])
    except Exception:
        pass

    print("=== Scene Setup (Blender) ===")
    print(f"Imported objects: {len(imported)}")
    print(f"Bounding box dims (m): {bbox_dims[0]:.6f}, {bbox_dims[1]:.6f}, {bbox_dims[2]:.6f}")
    print(f"Approx bounding-sphere radius (m): {bs_radius:.6f}")
    print(f"Engine: {args.engine}")
    print(f"Camera: {cam_obj.name} with target {cam_tgt.name}")
    print(f"Sun: {sun_obj.name}")
    print("Scene configured (engine, camera, sun). Compositor built.")

    # Ensure camera target sits at model center so the object stays centered
    center = get_model_center(bpy, imported)
    try:
        cam_tgt.location = center
    except Exception:
        pass

    # Precompute viewing directions and camera parameters for N samples
    dirs = fibonacci_sphere_directions(args.n, upper_hemisphere=True)
    rng = random.Random(args.seed)
    # Compute a base distance from the bounding box (tighter than sphere fit)
    fovx, fovy = get_camera_fov_xy(bpy, cam_obj, bpy.context.scene)
    base_bbox = compute_fit_distance_for_bbox(bbox_dims[0], bbox_dims[1], fovx, fovy, margin=1.00)
    distances, rolls = generate_camera_params(args.n, bs_radius, rng, cam_obj=cam_obj, base_override=base_bbox)
    split_seq = build_split_sequence(train_count, val_count, test_count)

    # Compute minimal distances so the object fits the frame (use bbox fit)
    fit_distance = compute_fit_distance_for_bbox(bbox_dims[0], bbox_dims[1], fovx, fovy, margin=0.50)
    base_dist = max(0.02, fit_distance * float(getattr(args, 'zoom', 0.80)))

    # Cache bbox points for visibility checks
    bbox_pts = all_world_bbox_points(bpy, imported)

    # Brief preview
    print("=== Render Plan Preview ===")
    print(f"Directions: {len(dirs)} | Distances: {len(distances)} | Rolls: {len(rolls)}")
    print(f"Split order: train x{train_count}, val x{val_count}, test x{test_count}")
    for i in range(min(3, args.n)):
        d = dirs[i]
        print(f"  [{i:03d}] split={split_seq[i]} dir=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) dist={distances[i]:.3f} roll={math.degrees(rolls[i]):+.1f} deg")
    print("Next step: implement render loop with explicit filenames and background swap.")

    # ===== Render Loop =====
    scn = bpy.context.scene
    nt = comp_refs['tree']
    nodes = comp_refs['nodes']
    n_out_rgb = nodes['out_rgb']
    n_out_mask = nodes['out_mask']
    n_out_depth = nodes['out_depth']

    # No background compositing; outputs are directly from Render Layers.

    # Initialize per-split counters
    split_counters = {"train": 1, "val": 1, "test": 1}
    total_done = 0
    current_img = None  # type: ignore

    for i in range(args.n):
        split = split_seq[i]
        idx_in_split = split_counters[split]
        frame_num = idx_in_split  # 1-based

        # Set File Output base paths to subfolders and 6-digit names
        n_out_rgb.base_path = dir_map[split]['rgb']
        n_out_mask.base_path = dir_map[split]['mask']
        n_out_depth.base_path = dir_map[split]['depth']
        try:
            if n_out_rgb.file_slots:
                n_out_rgb.file_slots[0].path = 'img_######'
            if n_out_mask.file_slots:
                n_out_mask.file_slots[0].path = 'img_######'
            if n_out_depth.file_slots:
                n_out_depth.file_slots[0].path = 'img_######'
        except Exception:
            pass

        # Set frame to control ###### digits
        scn.frame_set(frame_num)

        # No background: Film Transparent provides alpha; nothing to swap.

        # Keep camera target at model center every frame (safety)
        try:
            cam_tgt.location = center
        except Exception:
            pass

        # Camera placement
        dir_vec = dirs[i]
        dist_rand = distances[i]
        # Apply zoom as an upper cap (closer than per-sample if needed),
        # and min_fit as a lower floor to avoid clipping/empty frames.
        dist_cap = base_dist
        dist_floor = float(getattr(args, 'min_fit', 0.80)) * fit_distance
        dist = min(dist_rand, dist_cap)
        dist = max(dist, dist_floor)
        # Always keep camera outside the model's bounding sphere
        dist = max(dist, 1.05 * bs_radius)
        cam_obj.location = (dir_vec[0] * dist, dir_vec[1] * dist, dir_vec[2] * dist)
        if i < 5:
            print(f"[DBG] i={i} split={split} dist_rand={dist_rand:.4f} dist_cap={dist_cap:.4f} dist_floor={dist_floor:.4f} final_dist={dist:.4f}")
        # Keep clip distances safe
        try:
            cam_data = cam_obj.data
            cam_data.clip_start = 0.01
            cam_data.clip_end = max(cam_data.clip_end, dist + 3.0 * bs_radius)
        except Exception:
            pass
        # Apply small roll
        try:
            cam_obj.rotation_euler[2] = rolls[i]
        except Exception:
            pass

        # Sun randomization per sample (keep generally above and bright)
        try:
            # Keep sun mostly overhead with small jitter
            # Lower sun to get longer, more visible shadows on the ground
            base_rx = math.radians(40.0)
            base_ry = 0.0
            base_rz = math.radians(30.0)
            jx = math.radians(rng.uniform(-8.0, 8.0))
            jy = math.radians(rng.uniform(-5.0, 5.0))
            jz = math.radians(rng.uniform(-20.0, 20.0))
            sun_obj.rotation_euler = (base_rx + jx, base_ry + jy, base_rz + jz)
            # Random strength between ~5–7 for consistent shadows
            if hasattr(sun_obj.data, 'energy'):
                sun_obj.data.energy = rng.uniform(5.0, 7.0)
        except Exception:
            pass

        # Final framing: adjust to target fill and ensure visibility
        try:
            # Target fill: user override via --fill, else default 0.88
            target_fill = float(args.fill) if getattr(args, 'fill', None) is not None else 0.88
            # Clamp to sensible range
            if not (0.4 <= target_fill <= 0.99):
                target_fill = max(0.50, min(0.98, target_fill))
            dist_adj = adjust_distance_to_fill(
                bpy, scn, cam_obj, dir_vec, dist, bbox_pts, target_fill,
                fit_distance, bs_radius, float(getattr(args, 'min_fit', 0.80)), iters=4
            )
            cam_obj.location = (dir_vec[0] * dist_adj, dir_vec[1] * dist_adj, dir_vec[2] * dist_adj)
            # If still not visible, push out moderately
            if not is_bbox_in_view(bpy, scn, cam_obj, bbox_pts, margin=0.02):
                dist_safe = max(dist_adj, 1.00 * fit_distance, 1.05 * bs_radius)
                cam_obj.location = (dir_vec[0] * dist_safe, dir_vec[1] * dist_safe, dir_vec[2] * dist_safe)
        except Exception:
            pass

        # Render; File Output node writes the three outputs
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as ex:
            eprint(f"Render failed at sample {i+1}/{args.n}: {ex}")
            # Continue to next sample

        # No background image to free

        split_counters[split] += 1
        total_done += 1
        if (total_done % 25 == 0) or (total_done == args.n):
            print(f"Progress: {total_done}/{args.n} renders done...")


if __name__ == "__main__":
    main()
