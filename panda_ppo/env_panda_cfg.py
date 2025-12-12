import numpy as np
import mujoco as mj
from gymnasium import Env, spaces


class PandaEnv(Env):
    """
    Franka Panda reaching environment using ΔEE Cartesian actions.

    Action space (5D):
        [dx, dy, dz, dyaw, grip_cmd]

    Observation space (20D):
        [q(7), dq(7), ee_pos(3), cube_pos(3)]
    """

    metadata = {"render_modes": ["rgb_array"]}

    # ============================================================
    # Init
    # ============================================================
    def __init__(
            self, 
            xml_path, 
            dt=0.02, 
            headless=True,
            w_reach=1.0, 
            w_grasp=1.0, 
            w_lift=1.0,
            w_transport=1.0, 
            w_success=1.0,
            w_collision=1.0, 
            w_drop=1.0,
        ):

        super().__init__()

        self.dt = dt
        self.xml_path = xml_path

        # ------------------ MuJoCo model ------------------
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self.headless = headless

        # Sites & bodies
        self.ee_site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "panda_ee")
        if self.ee_site == -1:
            raise RuntimeError("Missing site: panda_ee")

        self.cube_names = ["cube_red", "cube_green", "cube_blue", "cube_yellow", "cube_purple"]
        self.cube_body = None

        # --------------------------------------------------
        # Gym Spaces
        # --------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-0.03, -0.03, -0.03, -0.3, -1.0], dtype=np.float32),
            high=np.array([ 0.03,  0.03,  0.03,  0.3,  1.0], dtype=np.float32)
        )

        obs_dim = 20
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high)

        # State
        self.step_count = 0
        self.max_steps = 200
        self._prev_action = None

        # Table height (adjust to XML if needed)
        self.table_height = 0.40
        # --- Gripper joint IDs ---
        self.gripper_left = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.gripper_right = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "finger_joint2")

        if self.gripper_left == -1 or self.gripper_right == -1:
            raise RuntimeError("Gripper finger joints not found in the model!")

        self.w_reach = w_reach
        self.w_grasp = w_grasp
        self.w_lift = w_lift
        self.w_transport = w_transport
        self.w_success = w_success
        self.w_collision = w_collision
        self.w_drop = w_drop
    # ============================================================
    # Helpers
    # ============================================================

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site].copy()

    def _get_cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def _pick_random_cube(self):
        name = np.random.choice(self.cube_names)
        body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if body == -1:
            raise RuntimeError(f"Cube {name} not found")
        self.cube_body = body

    def _get_obs(self):
        q = self.data.qpos[:7]
        dq = self.data.qvel[:7]
        ee = self._get_ee_pos()
        cube = self._get_cube_pos()
        return np.concatenate([q, dq, ee, cube]).astype(np.float32)
    
    def _apply_gripper(self, grip_cmd):
        """
        grip_cmd in [-1, 1]
        > 0 → close gripper
        < 0 → open gripper
        """

        # Normalize grip speed
        speed = 0.02 * np.clip(grip_cmd, -1.0, 1.0)

        # Finger joint positions
        left = self.data.qpos[self.gripper_left]
        right = self.data.qpos[self.gripper_right]

        # Gripper limits (Panda default)
        min_close = 0.0       # fully closed
        max_open = 0.04       # fully open

        if grip_cmd > 0:  
            # Close: move both fingers inward
            left  = max(min_close, left  - speed)
            right = max(min_close, right - speed)
        else:
            # Open: move both fingers outward
            left  = min(max_open, left  + speed)
            right = min(max_open, right + speed)

        # Write back joint values
        self.data.qpos[self.gripper_left] = left
        self.data.qpos[self.gripper_right] = right

    # ============================================================
    # Δx → Δq (Jacobian IK)
    # ============================================================
    def _apply_action(self, action):
        dx, dy, dz, dyaw, grip_cmd = action

        # --------- Arm motion (Jacobian IK) ----------
        v = np.zeros(6)
        v[0] = dx / self.dt
        v[1] = dy / self.dt
        v[2] = dz / self.dt
        v[5] = dyaw / self.dt

        # Compute EE Jacobian
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_site)
        J = np.vstack([Jp, Jr])[:, :7]

        dq = np.linalg.pinv(J, rcond=1e-4) @ v
        dq = np.clip(dq, -1.0, 1.0)
        self.data.qvel[:7] = dq

        # --------- Gripper motion ----------
        self._apply_gripper(grip_cmd)

    # ============================================================
    # Reward Terms
    # ============================================================

    def _joint_limit_penalty(self):
        q = self.data.qpos[:7]
        lo = self.model.jnt_range[:7, 0]
        hi = self.model.jnt_range[:7, 1]

        margin = 0.15
        pen = 0.0
        for qi, ql, qh in zip(q, lo, hi):
            if qi < ql + margin:
                pen += (ql + margin - qi)
            if qi > qh - margin:
                pen += (qi - (qh - margin))
        return -2.0 * pen

    def _accel_penalty(self, action):
        if self._prev_action is None:
            self._prev_action = np.zeros_like(action)
        accel = np.sum((action - self._prev_action) ** 2)
        self._prev_action = action.copy()
        return -0.1 * accel

    def _obstacle_penalty(self):
        ee = self._get_ee_pos()
        dz = ee[2] - self.table_height
        if dz < 0.06:  # 6 cm safe zone
            return -5.0 * (0.06 - dz)
        return 0.0

    # ============================================================
    # Compute Reward
    # ============================================================
    def _compute_reward(self, action):
        ee = self._get_ee_pos()
        cube = self._get_cube_pos()

        dist = np.linalg.norm(ee - cube)

        # ---------------- REACH SHAPING ----------------
        r_reach = 1.0 - np.tanh(dist)  # smoother than 1/dist
        r_reach *= 5.0

        # ---------------- SUCCESS ----------------------
        success = dist < 0.03
        r_success = 50.0 if success else 0.0

        # ---------------- PENALTIES --------------------
        r_action = -0.001 * np.sum(action**2)
        r_joint  = self._joint_limit_penalty()
        r_accel  = self._accel_penalty(action)
        r_obst   = self._obstacle_penalty()

        reward = (
            self.w_reach * r_reach +
            self.w_success * r_success +
            self.w_collision * r_obst +
            self.w_drop * r_action +
            self.w_grasp * r_joint +     
            self.w_lift * r_accel        
        )

        return float(reward), success, {
            "rew": {
                "reach": r_reach,
                "success": r_success,
                "joint_penalty": r_joint,
                "accel_penalty": r_accel,
                "obstacle_penalty": r_obst,
            },
            "success": success
        }   

    # ============================================================
    # Gym API
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        self._pick_random_cube()
        self.step_count = 0
        self.lifted = False

        # DEBUG: print cube info
        cube_name = self.model.names[self.model.name_bodyadr[self.cube_body]]
        cube_pos  = self.data.xpos[self.cube_body]
        # print("\n===== NEW EPISODE =====")
        # print(f"[RESET] Cube ID: {self.cube_body}")
        # print(f"[RESET] Cube Name: {cube_name}")
        # print(f"[RESET] Cube Position: {cube_pos}\n")
        return self._get_obs(), {}


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        # if self.step_count == 1:
        #     print(f"[EPISODE START] Cube: {self.cube_body}, Pos: {self.data.xpos[self.cube_body]}")

        # MuJoCo integration
        sub = max(1, int(self.dt / self.model.opt.timestep))
        for _ in range(sub):
            mj.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success, info = self._compute_reward(action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    # ============================================================
    # Rendering
    # ============================================================
    def render_frame(self):
        renderer = mj.Renderer(self.model, 480, 640)
        renderer.update_scene(self.data)
        return renderer.render()
