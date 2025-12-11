import numpy as np
import mujoco as mu
from gymnasium import Env, spaces


class PandaEnv(Env):
    """
    Franka Panda MuJoCo environment for pick-and-place with ΔEE control.

    Action (5):
        [dx, dy, dz, dyaw, gripper_cmd]  (Cartesian delta + yaw rate + gripper open/close)

    Observation (27):
        [ q(7), dq(7),
          ee_pos(3), ee_yaw(1),
          cube_pos(3), cube_yaw(1),
          goal_pos(3), goal_yaw(1),
          gripper_state(1) ]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml_path, dt=0.02, headless=True,
             w_reach=1.0, w_grasp=1.0, w_lift=1.0,
             w_transport=1.0, w_success=1.0,
             w_collision=1.0, w_drop=1.0):

        super().__init__()

        # --- Core sim ---
        self.model = mu.MjModel.from_xml_path(xml_path)
        self.data = mu.MjData(self.model)
        self.dt = float(dt)
        self.headless = bool(headless)

        # --- Renderer for video/logging (MuJoCo 3.x) ---
        # If camera name doesn't exist, we fall back to default free camera.
        self.renderer = mu.Renderer(self.model, height=480, width=640)
        self.camera_name = "table_rgb"
        cam_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.camera_id = cam_id if cam_id != -1 else None

        # --- Key sites & bodies ---
        self.ee_site = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_SITE, "panda_ee")
        if self.ee_site == -1:
            raise RuntimeError("Site 'panda_ee' not found in model.")
        self.goal_site = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_SITE, "goal")
        if self.goal_site == -1:
            raise RuntimeError("Site 'goal' not found in model.")

        self.cube_names = ["cube_red", "cube_green", "cube_blue", "cube_yellow", "cube_purple"]
        self.cube_body_id = None  # set on reset()

        # --- Finger joints (IDs) & DOF/QPOS indices ---
        self.lf_joint = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.rf_joint = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_JOINT, "finger_joint2")
        if self.lf_joint == -1 or self.rf_joint == -1:
            raise RuntimeError("Finger joints 'finger_joint1'/'finger_joint2' not found.")

        # map joint -> dof and qpos addresses
        self.lf_dof = int(self.model.jnt_dofadr[self.lf_joint])
        self.rf_dof = int(self.model.jnt_dofadr[self.rf_joint])
        self.lf_qpos = int(self.model.jnt_qposadr[self.lf_joint])
        self.rf_qpos = int(self.model.jnt_qposadr[self.rf_joint])

        # --- Action/Obs spaces ---
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.02, -0.02, -0.25, -1.0], dtype=np.float32),
            high=np.array([ 0.02,  0.02,  0.02,  0.25,  1.0], dtype=np.float32),
        )
        high = np.inf * np.ones(27, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # --- Episode state ---
        self.step_count = 0
        self.max_steps = 250
        self.lifted = False

        # Weights
        self.w_reach = w_reach
        self.w_grasp = w_grasp
        self.w_lift = w_lift
        self.w_transport = w_transport
        self.w_success = w_success
        self.w_collision = w_collision
        self.w_drop = w_drop

    # ============================================================
    # -------------------- Utility Functions ---------------------
    # ============================================================

    def _get_ee_pose(self):
        pos = self.data.site_xpos[self.ee_site].copy()
        mat_flat = self.data.site_xmat[self.ee_site].copy()  # len 9
        mat = mat_flat.reshape(3, 3)
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        return pos, yaw

    def _get_cube_pose(self):
        # cube_body_id is a BODY id; use body frames (xpos/xmat)
        pos = self.data.xpos[self.cube_body_id].copy()
        mat_flat = self.data.xmat[self.cube_body_id].copy()  # len 9
        mat = mat_flat.reshape(3, 3)
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        return pos, yaw

    def _get_goal_pose(self):
        pos = self.data.site_xpos[self.goal_site].copy()
        yaw = 0.0
        return pos, yaw

    def _get_gripper_state(self):
        # Sum of finger joint qpos (proxy for opening/closing)
        lf = float(self.data.qpos[self.lf_qpos])
        rf = float(self.data.qpos[self.rf_qpos])
        return lf + rf

    def _pick_random_cube(self):
        name = np.random.choice(self.cube_names)
        body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise RuntimeError(f"Cube body '{name}' not found.")
        self.cube_body_id = body_id

    # ============================================================
    # ----------------------- Observation -------------------------
    # ============================================================

    def _get_obs(self):
        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()

        ee_pos, ee_yaw = self._get_ee_pose()
        cube_pos, cube_yaw = self._get_cube_pose()
        goal_pos, goal_yaw = self._get_goal_pose()
        grip = np.array([self._get_gripper_state()], dtype=np.float32)

        obs = np.concatenate([
            q, dq,
            ee_pos, [ee_yaw],
            cube_pos, [cube_yaw],
            goal_pos, [goal_yaw],
            grip
        ]).astype(np.float32)
        return obs

    # ============================================================
    # --------------- ACTION → Δq via Jacobian -------------------
    # ============================================================

    def _apply_action(self, action):
        dx, dy, dz, dyaw, grip_cmd = action

        # Desired EE spatial velocity (cartesian)
        v = np.zeros(6, dtype=np.float64)
        v[:3] = np.array([dx, dy, dz], dtype=np.float64) / self.dt
        v[5] = float(dyaw) / self.dt

        # Jacobian at EE site
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mu.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_site)
        J = np.vstack([Jp, Jr])[:, :7]  # use 7 arm DOFs only

        # Resolved-rate IK: dq = pinv(J) * v
        dq = np.linalg.pinv(J, rcond=1e-4) @ v
        dq = np.clip(dq, -1.0, 1.0)

        # Apply joint velocities to arm
        self.data.qvel[:7] = dq

        # Gripper control (set finger DOF velocities)
        self._apply_gripper(grip_cmd)

    def _apply_gripper(self, cmd):
        # Close (>0) or open (<0) gripper by assigning DOF velocities
        if cmd > 0.0:
            # close
            self.data.qvel[self.lf_dof] = -0.2
            self.data.qvel[self.rf_dof] =  0.2
        else:
            # open
            self.data.qvel[self.lf_dof] =  0.2
            self.data.qvel[self.rf_dof] = -0.2

    # ============================================================
    # ---------------------- Reward Function ----------------------
    # ============================================================

    def _compute_reward(self, action):
        ee_pos, _ = self._get_ee_pose()
        cube_pos, _ = self._get_cube_pose()
        goal_pos, _ = self._get_goal_pose()

        dx = np.linalg.norm(ee_pos - cube_pos)
        dg = np.linalg.norm(cube_pos - goal_pos)

        # ----------------------------------------------------
        # 1) REACH REWARD (positive when approaching cube)
        # ----------------------------------------------------
        r_reach = 1.0 / (dx + 1e-6)        # closer = bigger reward
        r_reach = np.clip(r_reach, 0, 10)

        # ----------------------------------------------------
        # 2) GRASP REWARD
        # ----------------------------------------------------
        touching = dx < 0.04                    # near cube
        grip_close = action[4] > 0.0            # closing gripper

        # reward only when first time grasping
        if touching and grip_close:
            r_grasp = 5.0
        else:
            r_grasp = 0.0

        # ----------------------------------------------------
        # 3) LIFT REWARD (positive proportional to height)
        # ----------------------------------------------------
        cube_height = cube_pos[2]
        r_lift = 10.0 * cube_height

        # ----------------------------------------------------
        # 4) TRANSPORT REWARD (only if cube is lifted)
        # ----------------------------------------------------
        if cube_height > 0.03:
            r_transport = 1.0 / (dg + 1e-6)
            r_transport = np.clip(r_transport, 0, 10)
        else:
            r_transport = 0.0

        # ----------------------------------------------------
        # 5) GOAL SUCCESS REWARD
        # ----------------------------------------------------
        at_goal = (dg < 0.03 and cube_height > 0.05)
        r_success = 100.0 if at_goal else 0.0

        # ----------------------------------------------------
        # 6) TABLE COLLISION PENALTY
        # ----------------------------------------------------
        r_collision = 0.0
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            geom1 = self.model.geom_bodyid[contact.geom1]
            geom2 = self.model.geom_bodyid[contact.geom2]

            # Body names (from XML)
            name1 = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_BODY, geom1)
            name2 = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_BODY, geom2)


            if ("table" in name1 and "panda" in name2) or \
               ("table" in name2 and "panda" in name1):
                r_collision -= 10.0

        # ----------------------------------------------------
        # 7) DROPPING PENALTY
        # ----------------------------------------------------
        # If cube was lifted and then falls down → big penalty
        if self.lifted and cube_height < 0.02:
            r_drop = -20.0
        else:
            r_drop = 0.0

        # update lifted flag
        if cube_height > 0.04:
            self.lifted = True

        # ----------------------------------------------------
        # 8) ACTION REGULARIZATION
        # ----------------------------------------------------
        r_action = -0.001 * np.sum(action ** 2)

        # ----------------------------------------------------
        # TOTAL REWARD
        # ----------------------------------------------------
        reward = (
            self.w_reach * r_reach +
            self.w_grasp * r_grasp +
            self.w_lift * r_lift +
            self.w_transport * r_transport +
            self.w_success * r_success +
            self.w_collision * r_collision +
            self.w_drop * r_drop +
            r_action
        )

        return float(reward), at_goal, {
            "reach": float(r_reach),
            "grasp": float(r_grasp),
            "lift": float(r_lift),
            "transport": float(r_transport),
            "success": float(r_success),
            "collision": float(r_collision),
            "drop": float(r_drop),
        }




    # ============================================================
    # ---------------------- Gym API ------------------------------
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mu.mj_resetData(self.model, self.data)
        mu.mj_forward(self.model, self.data)  # ensure positions/orientations valid

        self._pick_random_cube()
        self.step_count = 0
        self.lifted = False

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)

        # Simulate at MuJoCo internal timestep to match dt
        substeps = int(np.round(self.dt / self.model.opt.timestep))
        for _ in range(max(substeps, 1)):
            mu.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success, rew_dict = self._compute_reward(action)

        self.step_count += 1
        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, {
            "success": success,
            "rew": rew_dict,
        }

    # ============================================================
    # --------------------- Rendering Utils -----------------------
    # ============================================================

    def render_frame(self):
        """Return an RGB frame (H, W, 3) uint8; works headless with mujoco.Renderer."""
        if self.camera_id is not None:
            self.renderer.update_scene(self.data, camera=self.camera_id)
        else:
            self.renderer.update_scene(self.data)  # default free camera
        frame = self.renderer.render()
        return frame  # (H, W, 3), uint8
