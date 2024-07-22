import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
import copy

script_dir = Path(__file__).resolve().parent
xml_path = str(script_dir / "franka_emika_panda" / "scene.xml")

# Levenberg-Marquardt IK solver
class IK_Solver():

    def __init__(self, model, site_name, Knull, workspace, q0, n_steps = 50, step_size = .1, tol = 1e-10, damping = 1e-3, gravity_compensation = True):
        self.model = copy.deepcopy(model)

        self.n_steps = n_steps
        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        # Whether to enable gravity compensation.
        self.gravity_compensation = gravity_compensation
        # End-effector site we wish to control
        self.site_id = self.model.site(site_name).id
        self.Knull = Knull
        self.workspace = workspace
        self.q0 = q0
    
    def solve(self, data, goal_pos, goal_quat):
        
        goal_pos = self.workspace.clip_to_workspace(goal_pos)

        self.data = copy.deepcopy(data)
        # Enable gravity compensation
        self.model.body_gravcomp[:] = float(self.gravity_compensation)

        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self.model.nv))
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        error_pos[:] = goal_pos - self.data.site(self.site_id).xpos
        mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        # quaternion that would transform from site to goal orientation
        mujoco.mju_mulQuat(error_quat, goal_quat, site_quat_conj)
        # velocity to get from site to goal orientation in 1 second
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
        n_steps = 0

        while (np.linalg.norm(error) >= self.tol and n_steps < self.n_steps):
            #calculate jacobian
            mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
            #calculate delta of joint q
            n = jac.shape[1]
            I = np.identity(n)
            product = jac.T @ jac + self.damping * I
            
            j_inv = np.linalg.pinv(product) @ jac.T
            #print(self.q0 - self.data.qpos)
            delta_q = j_inv @ error 
            delta_q += (np.identity(n) - np.linalg.pinv(jac) @ jac) @ (self.Knull * (self.q0 - self.data.qpos))
            #compute next step
            self.data.qpos += self.step_size * delta_q
            np.clip(self.data.qpos, *self.model.jnt_range.T, out=self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #recompute error
            error_pos[:] = goal_pos - self.data.site(self.site_id).xpos
            mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, goal_quat, site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            n_steps += 1
        
        if n_steps == self.n_steps:
            print(np.linalg.norm(error[:3]), np.linalg.norm(error[3:]))

        return self.data.qpos.copy()
    
class Robot_Workspace:
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.center = kwargs.get("center")
        if self.shape == "sphere":
            self.radius = kwargs.get("radius")
        elif self.shape == "box":
            self.width = kwargs.get("width")
            self.length = kwargs.get("length")
            self.height = kwargs.get("height")
    def clip_to_workspace(self, point):
        if self.shape == "sphere":
            if np.linalg.norm(point - self.center) <= self.radius:
                return point
            else:
                return self.center + (point - self.center) / np.linalg.norm(point - self.center) * self.radius