#!/usr/bin/env python3
"""
Generate PHUMA configuration files (custom.xml, scene.xml, config.yaml)
for a new humanoid robot from an input MuJoCo XML or URDF file.

Usage:
    python src/utils/setup_humanoid.py --input <robot.xml or robot.urdf> --humanoid_type <name>

This script:
1. Parses the input robot model
2. Finds left/right ankle_roll_link bodies
3. Computes toe/heel keypoint positions from foot mesh/geom extents
4. Generates custom.xml with keypoints added
5. Generates scene.xml
6. Generates config.yaml with bone mapping, keypoints, joints, etc.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PHUMA humanoid configuration files"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input MuJoCo XML or URDF file path"
    )
    parser.add_argument(
        "--humanoid_type", required=True,
        help="Name of the humanoid type (used as folder name)"
    )
    parser.add_argument(
        "--project_root", default=None,
        help="PHUMA project root directory (default: auto-detect)"
    )
    parser.add_argument(
        "--left_foot_body", default=None,
        help="Override left foot body name (default: auto-detect *ankle*roll*)"
    )
    parser.add_argument(
        "--right_foot_body", default=None,
        help="Override right foot body name (default: auto-detect *ankle*roll*)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# XML Utilities
# ---------------------------------------------------------------------------

def find_foot_bodies(root):
    """Find left and right foot body names (ankle_roll_link or similar)."""
    left_foot = None
    right_foot = None

    patterns_left = [
        r"left.*ankle.*roll",
        r"l_ankle_roll",
        r"left.*foot",
        r"l_foot",
    ]
    patterns_right = [
        r"right.*ankle.*roll",
        r"r_ankle_roll",
        r"right.*foot",
        r"r_foot",
    ]

    for body in root.iter("body"):
        name = body.get("name", "")
        if not left_foot:
            for p in patterns_left:
                if re.search(p, name, re.IGNORECASE):
                    left_foot = name
                    break
        if not right_foot:
            for p in patterns_right:
                if re.search(p, name, re.IGNORECASE):
                    right_foot = name
                    break

    return left_foot, right_foot


def compute_foot_extents(model, foot_body_name):
    """
    Compute toe and heel positions in the foot body's local frame
    by analyzing geom bounding boxes (mesh vertices, sphere, box, cylinder).
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_body_name)
    if body_id < 0:
        raise ValueError(f"Body '{foot_body_name}' not found in model")

    min_x, max_x = float("inf"), float("-inf")
    min_z = float("inf")
    found_geom = False

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue

        found_geom = True
        geom_type = model.geom_type[geom_id]
        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]
        geom_size = model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]

            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, geom_quat)
            rot = rot.reshape(3, 3)

            for v in vertices:
                v_body = rot @ v + geom_pos
                min_x = min(min_x, v_body[0])
                max_x = max(max_x, v_body[0])
                min_z = min(min_z, v_body[2])

        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = geom_size[0]
            min_x = min(min_x, geom_pos[0] - r)
            max_x = max(max_x, geom_pos[0] + r)
            min_z = min(min_z, geom_pos[2] - r)

        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = geom_size[:3]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, geom_quat)
            rot = rot.reshape(3, 3)
            # 8 corners of the box
            for dx in [-sx, sx]:
                for dy in [-sy, sy]:
                    for dz in [-sz, sz]:
                        corner = rot @ np.array([dx, dy, dz]) + geom_pos
                        min_x = min(min_x, corner[0])
                        max_x = max(max_x, corner[0])
                        min_z = min(min_z, corner[2])

        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = geom_size[0]
            min_x = min(min_x, geom_pos[0] - r)
            max_x = max(max_x, geom_pos[0] + r)
            min_z = min(min_z, geom_pos[2] - r)

    if not found_geom:
        raise ValueError(f"No geoms found for body '{foot_body_name}'")

    toe_pos = np.array([max_x, 0.0, min_z])
    heel_pos = np.array([min_x, 0.0, min_z])
    return toe_pos, heel_pos


def _extract_joint_info_from_xml(root):
    """Extract joint names, axes (dof), and ranges from an XML tree for T-pose computation."""
    joint_names = []
    joint_dofs = []
    joint_ranges = []

    def _traverse(body_elem):
        # Note: <freejoint> tags are skipped (no axis/range to extract)
        for joint in body_elem.findall("joint"):
            jtype = joint.get("type", "hinge")
            if jtype == "free":
                continue
            jname = joint.get("name", "")
            joint_names.append(jname)

            axis_str = joint.get("axis", "0 0 1")
            axis = [float(x) for x in axis_str.split()]
            abs_axis = [abs(a) for a in axis]
            joint_dofs.append(abs_axis.index(max(abs_axis)))

            range_str = joint.get("range", "")
            if range_str:
                lo, hi = [float(x) for x in range_str.split()]
                joint_ranges.append((lo, hi))
            else:
                joint_ranges.append((-3.14, 3.14))

        for child in body_elem.findall("body"):
            _traverse(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            _traverse(top_body)

    return joint_names, joint_dofs, joint_ranges


def _compute_min_world_z(model, data):
    """Compute the global minimum z across all geoms in world frame after mj_forward."""
    import mujoco

    min_world_z = float("inf")

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        geom_pos_world = data.geom_xpos[geom_id]
        geom_mat_world = data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
            for v in vertices:
                v_world = geom_mat_world @ v + geom_pos_world
                min_world_z = min(min_world_z, v_world[2])

        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = geom_size[0]
            min_world_z = min(min_world_z, geom_pos_world[2] - r)

        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = geom_size[:3]
            for dx in [-sx, sx]:
                for dy in [-sy, sy]:
                    for dz in [-sz, sz]:
                        corner = geom_mat_world @ np.array([dx, dy, dz]) + geom_pos_world
                        min_world_z = min(min_world_z, corner[2])

        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = geom_size[0]
            h = geom_size[1]
            for local_end in [np.array([0, 0, -h]), np.array([0, 0, h])]:
                end_world = geom_mat_world @ local_end + geom_pos_world
                min_world_z = min(min_world_z, end_world[2] - r)

        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = geom_size[0]
            h = geom_size[1]
            for local_end in [np.array([0, 0, -h]), np.array([0, 0, h])]:
                end_world = geom_mat_world @ local_end + geom_pos_world
                min_world_z = min(min_world_z, end_world[2] - r)

    return min_world_z


def compute_ground_adjusted_root_z(xml_path, original_root_z, tpose_dof_pos=None):
    """
    Compute the adjusted root_pos z so the robot's lowest point (feet)
    sits exactly at ground level (z=0).

    Sets the T-pose (dof_pos) before running mj_forward so the height
    is computed from the actual default pose, not the zero pose.

    1. Load model in MuJoCo
    2. Set T-pose qpos (root_pos + root_ori + dof_pos)
    3. Run mj_forward
    4. Find the global minimum z across all geoms (world frame)
    5. Adjust: new_root_z = original_root_z - min_world_z
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set T-pose before mj_forward (like visualize_toe_positions.py)
    qpos = np.zeros(model.nq)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    has_free = any("floating_base" in jn or model.jnt_type[i] == 0
                   for i, jn in enumerate(joint_names))

    if has_free:
        # floating base: qpos = [pos(3), quat(4), dof_pos(...)]
        qpos[0:3] = [0.0, 0.0, original_root_z]
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w,x,y,z identity quat
        if tpose_dof_pos is not None:
            qpos[7:7 + len(tpose_dof_pos)] = tpose_dof_pos
    else:
        if tpose_dof_pos is not None:
            n = min(len(tpose_dof_pos), model.nq)
            qpos[:n] = tpose_dof_pos[:n]

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    min_world_z = _compute_min_world_z(model, data)

    if min_world_z == float("inf"):
        print("  [WARNING] No geoms found, cannot adjust root z")
        return original_root_z

    adjusted_z = original_root_z - min_world_z
    print(f"  Root z adjustment (T-pose): original={original_root_z:.4f}, "
          f"min_world_z={min_world_z:.4f}, adjusted={adjusted_z:.4f}")
    return adjusted_z


def get_joint_axis_dof(joint_elem):
    """Get DOF index from joint axis: x=0, y=1, z=2."""
    axis_str = joint_elem.get("axis", "0 0 1")
    axis = [float(x) for x in axis_str.split()]
    abs_axis = [abs(a) for a in axis]
    return abs_axis.index(max(abs_axis))


# ---------------------------------------------------------------------------
# Generate custom.xml
# ---------------------------------------------------------------------------

def generate_custom_xml(input_path, output_path, humanoid_type,
                        left_foot_override=None, right_foot_override=None):
    """Generate custom.xml: copy input XML and add toe/heel keypoints."""
    import mujoco

    is_urdf = input_path.lower().endswith(".urdf")

    if is_urdf:
        model = mujoco.MjModel.from_xml_path(input_path)
        tmp_xml = output_path + ".tmp"
        mujoco.mj_saveLastXML(tmp_xml, model)
        xml_path = tmp_xml
    else:
        xml_path = input_path

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find foot bodies
    if left_foot_override and right_foot_override:
        left_foot = left_foot_override
        right_foot = right_foot_override
    else:
        left_foot, right_foot = find_foot_bodies(root)

    if not left_foot or not right_foot:
        raise ValueError(
            f"Could not find left/right foot bodies. "
            f"Found left={left_foot}, right={right_foot}. "
            f"Use --left_foot_body and --right_foot_body to specify."
        )

    print(f"  Foot bodies: left={left_foot}, right={right_foot}")

    # Load model in MuJoCo to compute foot extents and root z
    model = mujoco.MjModel.from_xml_path(xml_path)

    left_toe, left_heel = compute_foot_extents(model, left_foot)
    right_toe, right_heel = compute_foot_extents(model, right_foot)

    print(f"  Left  toe={left_toe}, heel={left_heel}")
    print(f"  Right toe={right_toe}, heel={right_heel}")

    # Compute T-pose dof_pos, then refine with MuJoCo FK optimization
    jnames, jdofs, jranges = _extract_joint_info_from_xml(root)
    tpose_dof_pos = compute_tpose_dof_pos(jnames, jdofs, jranges)

    # Find pelvis body original z (check both <joint type="free"> and <freejoint>)
    pelvis_z_original = 1.0
    for body in root.iter("body"):
        has_free = any(j.get("type") == "free" for j in body.findall("joint"))
        has_freejoint = len(body.findall("freejoint")) > 0
        if has_free or has_freejoint:
            pos_str = body.get("pos", "0 0 1.0")
            pelvis_z_original = float(pos_str.split()[2])
            break

    # Refine T-pose with MuJoCo: straighten arms via FK optimization
    tpose_dof_pos = refine_tpose_with_mujoco(model, tpose_dof_pos, pelvis_z=pelvis_z_original)
    print(f"  T-pose refined (arms straightened)")

    # Compute root z with T-pose applied (like visualize_toe_positions.py)
    adjusted_root_z = compute_ground_adjusted_root_z(
        xml_path, pelvis_z_original, tpose_dof_pos=tpose_dof_pos
    )

    # Check which keypoints already exist
    existing_keypoints = set()
    for body in root.iter("body"):
        name = body.get("name", "")
        if name in ("left_heel_keypoint", "left_toe_keypoint",
                     "right_heel_keypoint", "right_toe_keypoint"):
            existing_keypoints.add(name)

    # Add keypoints to the XML tree (skip if already present)
    for body in root.iter("body"):
        name = body.get("name", "")

        if name == left_foot:
            if "left_heel_keypoint" not in existing_keypoints:
                heel = ET.SubElement(body, "body")
                heel.set("name", "left_heel_keypoint")
                heel.set("pos", f"{left_heel[0]:.6f} {left_heel[1]:.6f} {left_heel[2]:.6f}")
                heel.append(ET.Comment(' <geom type="sphere" size="0.02" rgba="1 0 0 0.8"/> '))

            if "left_toe_keypoint" not in existing_keypoints:
                toe = ET.SubElement(body, "body")
                toe.set("name", "left_toe_keypoint")
                toe.set("pos", f"{left_toe[0]:.6f} {left_toe[1]:.6f} {left_toe[2]:.6f}")
                toe.append(ET.Comment(' <geom type="sphere" size="0.02" rgba="0 0 1 0.8"/> '))

        elif name == right_foot:
            if "right_heel_keypoint" not in existing_keypoints:
                heel = ET.SubElement(body, "body")
                heel.set("name", "right_heel_keypoint")
                heel.set("pos", f"{right_heel[0]:.6f} {right_heel[1]:.6f} {right_heel[2]:.6f}")
                heel.append(ET.Comment(' <geom type="sphere" size="0.02" rgba="1 0 0 0.8"/> '))

            if "right_toe_keypoint" not in existing_keypoints:
                toe = ET.SubElement(body, "body")
                toe.set("name", "right_toe_keypoint")
                toe.set("pos", f"{right_toe[0]:.6f} {right_toe[1]:.6f} {right_toe[2]:.6f}")
                toe.append(ET.Comment(' <geom type="sphere" size="0.02" rgba="0 0 1 0.8"/> '))

    if existing_keypoints:
        print(f"  [INFO] Keypoints already exist, skipped: {existing_keypoints}")

    # Update model name
    root.set("model", humanoid_type)

    # Write with indentation
    ET.indent(tree, space="  ")
    tree.write(output_path, xml_declaration=False, encoding="unicode")

    # Clean up temp file
    if is_urdf:
        tmp_xml = output_path + ".tmp"
        if os.path.exists(tmp_xml):
            os.remove(tmp_xml)

    print(f"  -> Saved: {output_path}")
    return model, adjusted_root_z, tpose_dof_pos


# ---------------------------------------------------------------------------
# Generate scene.xml
# ---------------------------------------------------------------------------

def generate_scene_xml(output_path, humanoid_type):
    """Generate scene.xml with standard PHUMA scene setup."""
    content = f"""<mujoco model="{humanoid_type} scene">
  <include file="custom.xml"/>

  <statistic center="1.0 0.7 1.5" extent="0.8"/>

  <visual>
    <map force="0.1" fogend="20" fogstart="10" shadowclip="5" />
    <headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0.9 0.9 0.9"/>
    <global offwidth="2560" offheight="1920"
            azimuth="-140" elevation="-20"/>
    <quality shadowsize="16384"/>

  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.9176 0.9216 0.9294" rgb2="0.9176 0.9216 0.9294" width="100" height="100"/>
    <texture builtin="checker" height="100" width="100" name="texplane" rgb1="0.7490 0.8392 0.8353" rgb2="0.7490 0.8392 0.8353" type="2d" mark="edge" markrgb="1 1 1"/>
    <material name="MatPlane" reflectance="0.0" shininess="2" specular="1" texrepeat="120 120" texture="texplane"/>
  </asset>

  <worldbody>
    <light pos="3.5 0 3.5" dir="-1 0 -1" directional="true"/>
    <geom name="floor" size="30 20 0.05" type="plane" material="MatPlane"/>

  </worldbody>
</mujoco>
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"  -> Saved: {output_path}")


# ---------------------------------------------------------------------------
# Generate config.yaml
# ---------------------------------------------------------------------------

def _find_body_name(body_names, patterns):
    """Find first body name matching any pattern."""
    for name in body_names:
        for p in patterns:
            if re.search(p, name, re.IGNORECASE):
                return name
    return None


def _find_all_body_names(body_names, patterns):
    """Find all body names matching any pattern."""
    results = []
    for name in body_names:
        for p in patterns:
            if re.search(p, name, re.IGNORECASE):
                results.append(name)
                break
    return results


def compute_tpose_dof_pos(joint_names, joint_dofs, joint_ranges):
    """
    Compute initial joint positions for a T-pose (shoulder_roll only).
    Use refine_tpose_with_mujoco() afterwards for full arm straightening.
    """
    half_pi = np.pi / 2.0
    dof_pos = []

    for i, jname in enumerate(joint_names):
        lo, hi = joint_ranges[i]
        val = 0.0

        # Left shoulder roll joint → +π/2
        if re.search(r"left.*shoulder.*roll", jname, re.IGNORECASE):
            val = min(half_pi, hi)
            val = max(val, lo)
        # Right shoulder roll joint → -π/2
        elif re.search(r"right.*shoulder.*roll", jname, re.IGNORECASE):
            val = max(-half_pi, lo)
            val = min(val, hi)

        dof_pos.append(round(val, 4))

    return dof_pos


def refine_tpose_with_mujoco(model, dof_pos, pelvis_z=1.0):
    """
    Refine T-pose using MuJoCo FK + optimization.

    For each arm, optimizes shoulder_pitch/yaw, elbow, and wrist joint angles
    to maximize arm extension (straight arm) while keeping horizontal.
    shoulder_roll is kept fixed at ±π/2.
    """
    import mujoco
    from scipy.optimize import minimize

    data = mujoco.MjData(model)
    dof_pos = list(dof_pos)  # mutable copy

    has_free = model.njnt > 0 and model.jnt_type[0] == 0
    free_offset = 1 if has_free else 0  # joints to skip in indexing

    # Build base qpos
    base_qpos = np.zeros(model.nq)
    if has_free:
        base_qpos[0:3] = [0, 0, pelvis_z]
        base_qpos[3:7] = [1, 0, 0, 0]
        base_qpos[7:7 + len(dof_pos)] = dof_pos
    else:
        n = min(len(dof_pos), model.nq)
        base_qpos[:n] = dof_pos[:n]

    for side in ['left', 'right']:
        # Collect arm joints to optimize (exclude shoulder_roll which is fixed)
        arm_dof_indices = []  # index into dof_pos
        arm_bounds = []

        for ji in range(model.njnt):
            if model.jnt_type[ji] != 3:  # skip non-hinge (free joint etc.)
                continue
            jname = model.joint(ji).name
            if side not in jname.lower():
                continue

            is_arm = any(p in jname.lower() for p in ['shoulder', 'elbow', 'wrist'])
            is_fixed = 'shoulder' in jname.lower() and 'roll' in jname.lower()

            if is_arm and not is_fixed:
                dof_idx = ji - free_offset
                if 0 <= dof_idx < len(dof_pos):
                    arm_dof_indices.append(dof_idx)
                    lo, hi = model.jnt_range[ji]
                    arm_bounds.append((float(lo), float(hi)))

        if not arm_dof_indices:
            continue

        # Find shoulder_roll body and last wrist body
        shoulder_bid = None
        wrist_bid = None
        for bi in range(model.nbody):
            bname = model.body(bi).name.lower()
            if side in bname and 'shoulder' in bname and 'roll' in bname:
                shoulder_bid = bi
            if side in bname and ('wrist' in bname or 'hand' in bname or 'welder' in bname):
                wrist_bid = bi  # take the last match

        if shoulder_bid is None or wrist_bid is None:
            continue

        def objective(angles, _arm_indices=arm_dof_indices,
                      _s_bid=shoulder_bid, _w_bid=wrist_bid):
            qpos = base_qpos.copy()
            qpos_offset = 7 if has_free else 0
            for di, angle in zip(_arm_indices, angles):
                qpos[qpos_offset + di] = angle
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            s_pos = data.xpos[_s_bid]
            w_pos = data.xpos[_w_bid]

            # Maximize distance (arm extension)
            dist = np.linalg.norm(w_pos - s_pos)
            # Penalize vertical deviation (keep horizontal)
            z_penalty = (w_pos[2] - s_pos[2]) ** 2

            return -dist + 10.0 * z_penalty

        x0 = [dof_pos[di] for di in arm_dof_indices]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=arm_bounds)

        if result.fun < objective(x0):
            for di, angle in zip(arm_dof_indices, result.x):
                dof_pos[di] = round(float(angle), 4)

        # Update base_qpos for next arm
        qpos_offset = 7 if has_free else 0
        for di in arm_dof_indices:
            base_qpos[qpos_offset + di] = dof_pos[di]

    return dof_pos


def generate_config_yaml(xml_path, output_path, humanoid_type,
                         adjusted_root_z=None, tpose_dof_pos_refined=None):
    """Generate config.yaml from the robot model XML (the custom.xml with keypoints)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---- Traverse body tree to collect bodies, joints in order ----
    body_names = []
    joint_names = []
    joint_dofs = []
    joint_axes = []  # actual 3D axis vectors for each joint
    joint_vel_limits = []
    joint_ranges = []  # (low, high) for each joint
    free_joint_name = None
    pelvis_name = None
    pelvis_pos = [0.0, 0.0, 1.0]

    def traverse_body(body_elem):
        nonlocal free_joint_name, pelvis_name, pelvis_pos

        name = body_elem.get("name", "")
        if name:
            body_names.append(name)

        # Detect <freejoint> tag (e.g. <freejoint name="pelvis_link"/>)
        for fj in body_elem.findall("freejoint"):
            free_joint_name = "floating_base_joint"
            pelvis_name = name
            pos_str = body_elem.get("pos", "0 0 1.0")
            pelvis_pos = [float(x) for x in pos_str.split()]

        for joint in body_elem.findall("joint"):
            jname = joint.get("name", "")
            jtype = joint.get("type", "hinge")

            if jtype == "free":
                free_joint_name = "floating_base_joint"
                pelvis_name = name
                pos_str = body_elem.get("pos", "0 0 1.0")
                pelvis_pos = [float(x) for x in pos_str.split()]
            else:
                joint_names.append(jname)
                dof = get_joint_axis_dof(joint)
                joint_dofs.append(dof)
                # Store actual joint axis vector
                axis_str = joint.get("axis", "0 0 1")
                axis = [float(x) for x in axis_str.split()]
                norm = sum(a*a for a in axis) ** 0.5
                joint_axes.append([round(a / norm, 6) for a in axis])

                # Extract joint range
                range_str = joint.get("range", "")
                if range_str:
                    lo, hi = [float(x) for x in range_str.split()]
                    joint_ranges.append((lo, hi))
                else:
                    joint_ranges.append((-3.14, 3.14))

                # Extract velocity limit from actuator force range (heuristic)
                frcrange = joint.get("actuatorfrcrange", "")
                if frcrange:
                    forces = [abs(float(x)) for x in frcrange.split()]
                    max_force = max(forces)
                    joint_vel_limits.append(round(max_force / 5, 1))
                else:
                    joint_vel_limits.append(20.0)

        for child in body_elem.findall("body"):
            traverse_body(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            traverse_body(top_body)

    if not pelvis_name:
        pelvis_name = body_names[0] if body_names else "pelvis"

    # ---- Auto-adjust root_pos z so feet touch ground (z=0) ----
    if adjusted_root_z is not None:
        pelvis_pos[2] = adjusted_root_z

    # ---- Find key body names for bone mapping ----
    left_hip = _find_body_name(body_names, [r"^left.*hip.*roll"])
    left_knee = _find_body_name(body_names, [r"^left.*knee.*link$", r"^left.*knee"])
    left_ankle = _find_body_name(body_names, [r"^left.*ankle.*pitch"])
    right_hip = _find_body_name(body_names, [r"^right.*hip.*roll"])
    right_knee = _find_body_name(body_names, [r"^right.*knee.*link$", r"^right.*knee"])
    right_ankle = _find_body_name(body_names, [r"^right.*ankle.*pitch"])

    torso = _find_body_name(body_names, [r"^torso", r"waist_pitch"])
    left_shoulder = _find_body_name(body_names, [r"^left.*shoulder.*(roll|keypoint)"])
    left_elbow = _find_body_name(body_names, [r"^left.*elbow.*(link|keypoint|pitch)"])
    right_shoulder = _find_body_name(body_names, [r"^right.*shoulder.*(roll|keypoint)"])
    right_elbow = _find_body_name(body_names, [r"^right.*elbow.*(link|keypoint|pitch)"])

    # Last wrist link for each side
    left_wrists = _find_all_body_names(body_names, [r"^left.*wrist"])
    right_wrists = _find_all_body_names(body_names, [r"^right.*wrist"])
    left_wrist = left_wrists[-1] if left_wrists else None
    right_wrist = right_wrists[-1] if right_wrists else None

    # Ensure heel/toe keypoints are in body_names (add after ankle_roll if missing)
    left_foot_name = _find_body_name(body_names, [r"^left.*ankle.*roll"])
    right_foot_name = _find_body_name(body_names, [r"^right.*ankle.*roll"])

    # Deduplicate body_names (in case keypoints already existed in the XML)
    seen = set()
    deduped = []
    for bn in body_names:
        if bn not in seen:
            seen.add(bn)
            deduped.append(bn)
    body_names = deduped

    # Insert keypoints after foot bodies if not already present
    if left_foot_name and "left_heel_keypoint" not in seen:
        idx = body_names.index(left_foot_name) + 1
        body_names.insert(idx, "left_heel_keypoint")
        body_names.insert(idx + 1, "left_toe_keypoint")
    if right_foot_name and "right_heel_keypoint" not in seen:
        idx = body_names.index(right_foot_name) + 1
        body_names.insert(idx, "right_heel_keypoint")
        body_names.insert(idx + 1, "right_toe_keypoint")

    # ---- Build config string (manual formatting to match reference) ----
    lines = []

    # root_pos
    lines.append(f"root_pos: [{pelvis_pos[0]:.4f}, {pelvis_pos[1]:.4f}, {pelvis_pos[2]:.4f}]")
    lines.append("root_ori: [0.0000, 0.0000, 0.0000, 1.0000]")

    # dof_pos (T-pose: arms horizontal to sides, legs straight)
    if tpose_dof_pos_refined is not None:
        dof_pos = list(tpose_dof_pos_refined)
    else:
        dof_pos = compute_tpose_dof_pos(joint_names, joint_dofs, joint_ranges)
    lines.append("dof_pos: [")
    # Group by limbs based on joint name patterns
    current_group = []
    groups = []
    for i, jname in enumerate(joint_names):
        current_group.append(f"{dof_pos[i]:.4f}")
        # Start new group at boundaries between limbs
        if any(re.search(p, jname) for p in [
            r"ankle_roll", r"wrist_yaw", r"torso_joint",
            r"waist_pitch", r"elbow_joint$", r"elbow_roll",
        ]):
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)

    for i, group in enumerate(groups):
        sep = "," if i < len(groups) - 1 else ""
        indent_vals = ", ".join(f"{v:>8s}" for v in group)
        lines.append(f"  {indent_vals}{sep}")
        if i < len(groups) - 1:
            lines.append("")
    lines.append("]")
    lines.append("")

    # bone_mapping
    lines.append("bone_mapping:")
    lines.append("  # Left Leg")
    if left_hip:
        lines.append(f"  - ['pelvis', 'left_hip', '{pelvis_name}', '{left_hip}']")
    if left_hip and left_knee:
        lines.append(f"  - ['left_hip', 'left_knee', '{left_hip}', '{left_knee}']")
    if left_knee and left_ankle:
        lines.append(f"  - ['left_knee', 'left_ankle', '{left_knee}', '{left_ankle}']")
    lines.append("  # Right Leg")
    if right_hip:
        lines.append(f"  - ['pelvis', 'right_hip', '{pelvis_name}', '{right_hip}']")
    if right_hip and right_knee:
        lines.append(f"  - ['right_hip', 'right_knee', '{right_hip}', '{right_knee}']")
    if right_knee and right_ankle:
        lines.append(f"  - ['right_knee', 'right_ankle', '{right_knee}', '{right_ankle}']")
    lines.append("  # Left Arm")
    if torso and left_shoulder:
        lines.append(f"  - ['pelvis', 'left_shoulder', '{torso}', '{left_shoulder}']")
    if left_shoulder and left_elbow:
        lines.append(f"  - ['left_shoulder', 'left_elbow', '{left_shoulder}', '{left_elbow}']")
    if left_elbow and left_wrist:
        lines.append(f"  - ['left_elbow', 'left_wrist', '{left_elbow}', '{left_wrist}']")
    lines.append("  # Right Arm")
    if torso and right_shoulder:
        lines.append(f"  - ['pelvis', 'right_shoulder', '{torso}', '{right_shoulder}']")
    if right_shoulder and right_elbow:
        lines.append(f"  - ['right_shoulder', 'right_elbow', '{right_shoulder}', '{right_elbow}']")
    if right_elbow and right_wrist:
        lines.append(f"  - ['right_elbow', 'right_wrist', '{right_elbow}', '{right_wrist}']")
    lines.append("")

    # keypoints (fixed names)
    lines.append("keypoints:")
    kp_entries = [
        ("pelvis",                  pelvis_name),
        ("left_hip_keypoint",       left_hip or ""),
        ("left_knee_keypoint",      left_knee or ""),
        ("left_ankle_keypoint",     left_ankle or ""),
        ("left_heel_keypoint",      "left_heel_keypoint"),
        ("left_toe_keypoint",       "left_toe_keypoint"),
        ("right_hip_keypoint",      right_hip or ""),
        ("right_knee_keypoint",     right_knee or ""),
        ("right_ankle_keypoint",    right_ankle or ""),
        ("right_heel_keypoint",     "right_heel_keypoint"),
        ("right_toe_keypoint",      "right_toe_keypoint"),
        ("torso_keypoint",          torso or ""),
        ("left_shoulder_keypoint",  left_shoulder or ""),
        ("left_elbow_keypoint",     left_elbow or ""),
        ("left_wrist_keypoint",     left_wrist or ""),
        ("right_shoulder_keypoint", right_shoulder or ""),
        ("right_elbow_keypoint",    right_elbow or ""),
        ("right_wrist_keypoint",    right_wrist or ""),
    ]
    max_name_len = max(len(n) for n, _ in kp_entries)
    max_body_len = max(len(b) for _, b in kp_entries)
    for kp_name, kp_body in kp_entries:
        lines.append(
            f"  - {{ name: '{kp_name}',{' ' * (max_name_len - len(kp_name) + 1)}"
            f"body: '{kp_body}',{' ' * (max_body_len - len(kp_body) + 1)}}}"
        )
    lines.append("")

    # body_names
    lines.append("body_names: [")
    # Group by structure
    body_groups = []
    current_group = [body_names[0]]  # pelvis
    side_order = ["left_hip", "left_knee", "left_ankle", "left_heel", "left_toe",
                   "right_hip", "right_knee", "right_ankle", "right_heel", "right_toe",
                   "waist", "torso",
                   "left_shoulder", "left_elbow", "left_wrist",
                   "right_shoulder", "right_elbow", "right_wrist"]

    # Simple approach: just list all body names
    body_str_parts = []
    for i, bn in enumerate(body_names):
        sep = "," if i < len(body_names) - 1 else ""
        body_str_parts.append(f"  '{bn}'{sep}")
    lines.extend(body_str_parts)
    lines.append("]")
    lines.append("")

    # joint_names
    lines.append("joint_names: [")
    if free_joint_name:
        lines.append(f"  '{free_joint_name}',")
        lines.append("")
    for i, jn in enumerate(joint_names):
        sep = "," if i < len(joint_names) - 1 else ""
        lines.append(f"  '{jn}'{sep}")
    lines.append("]")
    lines.append("")

    # joint_velocity_limits
    lines.append("joint_velocity_limits: [")
    for i, vl in enumerate(joint_vel_limits):
        sep = "," if i < len(joint_vel_limits) - 1 else ""
        lines.append(f"  {vl}{sep}")
    lines.append("]")
    lines.append("")

    # dof
    lines.append("dof: [")
    if free_joint_name:
        lines.append("  2,")
        lines.append("")
    for i, d in enumerate(joint_dofs):
        sep = "," if i < len(joint_dofs) - 1 else ""
        lines.append(f"  {d}{sep}")
    lines.append("]")
    lines.append("")

    # joint_axes - actual 3D axis vectors for FK (handles non-axis-aligned joints)
    lines.append("joint_axes: [")
    if free_joint_name:
        lines.append("  [0, 0, 1],")
        lines.append("")
    for i, ax in enumerate(joint_axes):
        sep = "," if i < len(joint_axes) - 1 else ""
        lines.append(f"  [{ax[0]}, {ax[1]}, {ax[2]}]{sep}")
    lines.append("]")

    config_str = "\n".join(lines) + "\n"

    with open(output_path, "w") as f:
        f.write(config_str)

    print(f"  -> Saved: {output_path}")

    # Print warnings for missing mappings
    missing = []
    for label, val in [
        ("left_hip", left_hip), ("left_knee", left_knee),
        ("left_ankle", left_ankle), ("right_hip", right_hip),
        ("right_knee", right_knee), ("right_ankle", right_ankle),
        ("torso", torso), ("left_shoulder", left_shoulder),
        ("left_elbow", left_elbow), ("left_wrist", left_wrist),
        ("right_shoulder", right_shoulder), ("right_elbow", right_elbow),
        ("right_wrist", right_wrist),
    ]:
        if not val:
            missing.append(label)

    if missing:
        print(f"\n  [WARNING] Could not auto-detect these bodies: {missing}")
        print("  Please edit config.yaml manually to fill in the correct body names.")

    print("\n  [NOTE] Please review and adjust:")
    print("    - dof_pos: set proper default joint positions")
    print("    - joint_velocity_limits: verify values (currently estimated from actuator force range)")
    print("    - bone_mapping: verify body name mappings are correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Detect project root
    if args.project_root:
        project_root = args.project_root
    else:
        # Walk up from this script's location to find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    output_dir = os.path.join(project_root, "asset", "humanoid_model", args.humanoid_type)
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.abspath(args.input)
    custom_xml_path = os.path.join(output_dir, "custom.xml")
    scene_xml_path = os.path.join(output_dir, "scene.xml")
    config_yaml_path = os.path.join(output_dir, "config.yaml")

    print(f"=== Setting up humanoid: {args.humanoid_type} ===")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_dir}/")
    print()

    # 1. Generate custom.xml (with toe/heel keypoints)
    print("[1/3] Generating custom.xml ...")
    _, adjusted_root_z, tpose_dof_pos = generate_custom_xml(
        input_path, custom_xml_path, args.humanoid_type,
        left_foot_override=args.left_foot_body,
        right_foot_override=args.right_foot_body,
    )
    print()

    # 2. Generate scene.xml
    print("[2/3] Generating scene.xml ...")
    generate_scene_xml(scene_xml_path, args.humanoid_type)
    print()

    # 3. Generate config.yaml (parse from the custom.xml that now has keypoints)
    print("[3/3] Generating config.yaml ...")
    generate_config_yaml(custom_xml_path, config_yaml_path, args.humanoid_type,
                         adjusted_root_z=adjusted_root_z,
                         tpose_dof_pos_refined=tpose_dof_pos)
    print()

    print(f"=== Done! Files saved to {output_dir}/ ===")


if __name__ == "__main__":
    main()
