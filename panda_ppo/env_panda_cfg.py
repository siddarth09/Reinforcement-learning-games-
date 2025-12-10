import numpy as np 
import mujoco as mu 
from gymnasium import Env,spaces 

class PandaEnv(Env):
    """
    Franka panda arm mujoco environment
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,xml_path,dt=0.02):
        super().__init__()

        self.model = mu.MjModel.from_xml_path(xml_path)
        self.data = mu.MjData(self.model)
        self.dt = dt 


        self.ee_site = mu.mj_name2id(self.model,mu.mjtObj.mjOBJ_SITE,"panda_ee")
        self.goal_site= mu.mj_name2id(self.model,mu.mjtObj.mjOBJ_SITE,"goal")

        self.cube_names= [
            "cube_red",
            "cube_green",
            "cube_blue",
            "cube_yellow",
            "cube_purple",
        ]

        self.lf = mu.mj_name2id(self.model,mu.mjtObj.mjOBJ_JOINT,"finger_joint1")
        self.rf = mu.mj_name2id(self.model,mu.mjtObj.mjOBJ_JOINT,"finger_joint2")

        # Action space 

        self.action_space = spaces.Box(
            low = np.array([-0.02,-0.02,-0.02,-0.25,-1.0],dtype=np.float32),
            high= np.array([0.02,0.02,0.02,0.25,1.0],dtype=np.float32)
        )

        high = np.inf* np.ones(27,dtype=np.float32)
        self.observation_space= spaces.Box(-high,high,dtype=np.float32)

        # episode state 
        self.step_count = 0 
        self.max_steps = 250 
        self.lifted = False 
        self.cube_body_id = None 
        self.viewer = None 


    # ============================================================
    # -------------------- Utility Functions ---------------------
    # ============================================================

    def _get_ee_pose(self):
        pos = self.data.site_xpos[self.ee_site].copy()
        mat = self.data.site_xmat[self.ee_site].reshape(3,3)
        yaw = np.arctan2(mat[1,0],mat[0,0])

        print("Gripper Position = {0} and yaw = {1}".format(pos,yaw))
        return pos,yaw
    
    def _get_cube_pose(self):
        pos = self.data.site_xpos[self.cube_body_id].copy()
        mat = self.data.site_xmat[self.cube_body_id].reshape(3,3)
        yaw = np.arctan2(mat[1,0],mat[0,0])

        print("Cube Position = {0} and yaw = {1}".format(pos,yaw))
        return pos,yaw
    
    def _get_goal_pose(self):

        pos = self.data.site_xpos[self.goal_site].copy()
        yaw = 0.0 
        return pos,yaw 
    
    def _get_gripper_state(self):
        lf = self.data.qpos[self.lf]
        rf = self.data.qpos[self.rf]

        return lf+rf 
    
    def _pick_random_cube(self):

        name  = np.random.choice (self.cube_names)

        self.cube_body_id = mu.mj_name2id(self.model,mu.mjtObj.mjOBJ_SITE,name)

    # ============================================================
    # ----------------------- Observation -------------------------
    # ============================================================

    def _get_obs(self):

        q = self.data.qpos[:7].copy() 
        dq = self.data.qvel[:7].copy() 

        ee_pos, ee_yaw = self._get_ee_pose()
        cube_pos,cube_yaw = self._get_cube_pose() 
        goal_pos,goal_yaw = self._get_goal_pose() 
        grip = np.array([self._get_gripper_state()],dtype=np.float32)

        obs = np.concatenate([
            q,dq,
            ee_pos,[ee_yaw],
            cube_pos,[cube_yaw],
            goal_pos,[goal_yaw],
            grip
        ]).astype(np.float32)

    # ============================================================
    # --------------- ACTION → Δq via Jacobian -------------------
    # ============================================================

    def _apply_action(self,action):

        dx,dy,dz,dyaw,grip_cmd = action 

        v = np.zeros(6)
        v[:3] = np.array([dx,dy,dz])/self.dt 
        v[5] = dyaw / self.dt 

        # Jacobian 

        Jp = np.zeros((3,self.model.nv))
        Jr  = np.zeros((3,self.model.nv))
        mu.mj_jacSite(self.model,self.data,Jp,Jr,self.ee_site)
        J = np.vstack([Jp,Jr])[:,:7]

        # Pseudo-inverse 
        dq = np.linalg.pinv(J,rcond= 1e-4) @ v 
        dq = np.clip(dq,-1.0,1.0)

        self.data.qvel[:7] = dq 

        self._apply_gripper(grip_cmd)

    def _apply_gripper(self,cmd):

        if cmd > 0 : 
            self.data.qvel[self.lf] = -0.2
            self.data.qvel[self.rf] = 0.2 

        else: 

            self.data.qvel[self.lf] = 0.2 
            self.data.qvel[self.rf] = -0.2 

    # ============================================================
    # ---------------------- Reward Function ----------------------
    # ============================================================

    def _compute_reward(self,action):

        ee_pos,ee_yaw = self._get_ee_pose() 
        cube_pos,cube_yaw = self._get_cube_pose() 
        goal_pos,_ = self._get_goal_pose() 

        # Reaching reward 

        reach_dis = np.linalg.norm(ee_pos-cube_pos)
        r_reach = 1.0 * np.exp(-(reach_dis**2)/0.02)

        # Alignment Reward 

        yaw_err = abs(ee_yaw-cube_yaw)
        r_align = 0.5 * np.exp(-(yaw_err)/0.1)

        # Grasp Reward 

        touching = reach_dis < 0.04 
        closing  = action[4] >0 
        r_grasp = 2.0 if (touching and closing) else 0.0 

        # Lift reward 

        cube_height = cube_pos[2]
        h_threshold = 0.05 
        r_lift = 2.0 if cube_height > h_threshold else 0.0 


        # Place Reward 

        place_dist = np.linalg.norm(cube_pos - goal_pos)
        r_place = 4.0 * np.exp(-(place_dist**2) / 0.02)

        # ~~~~~~~~~~ Success Reward ~~~~~~~~~~
        success = (place_dist < 0.02 and cube_height > 0.03)
        r_success = 10.0 if success else 0.0

        # ~~~~~~~~~~ Penalties ~~~~~~~~~~
        r_action = -1e-3 * np.sum(action**2)

        reward = r_reach + r_align + r_grasp + r_lift + r_place + r_success + r_action

        return reward, success
    
    # ============================================================
    # ---------------------- Gym API ------------------------------
    # ============================================================

    def reset(self,seed= None,options = None):
        super().reset(seed=seed)
        mu.mj_resetData(self.model,self.data)

        self._pick_random_cube()
        self.step_count = 0 
        self.lifted = False 

        return self._get_obs(),{} 
    

    def step (self,action):

        action = np.clip (action,self.action_space.low,self.action_space.high)
        self._apply_action(action) 

        # Simulate 

        substeps = int(self.dt/self.model.opt.timestep) 
        for _ in range(max(substeps,1)):
            mu.mj_step(self.model,self.data) 

        obs = self._get_obs() 
        reward,success = self._compute_reward(action)

        self.step_count +=1 
        terminated = success 
        truncated = self.count>= self.max_steps 

        return obs,reward,terminated,truncated,{"success":success}
    

    def render(self):
        if self.viewer is None: 
            self.viewer = mu.viewer.launch_passive(self.model,self.data)

        self.viewer.sync() 


    def close(self):
        if self.viwer is not None: 
            self.viewer.close() 
            self.viewer = None 


            




