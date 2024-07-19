import numpy as np
import mujoco
import mujoco_viewer
import random
import time

class Sim:

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
        self.data = mujoco.MjData(self.model)
        self.initial_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        self.K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0] # gains for pd controller
        self.fps = 30
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.max_pos_delta = 1.1
        self.max_ori_delta = 1.1

    def set_pose(self, pose):
        for i in range(1, 8): 
            self.data.joint(f"panda_joint{i}").qpos = pose[i - 1]
        mujoco.mj_forward(self.model, self.data)

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_link7").xpos
        xquat = self.data.body("panda_link7").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_link7")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)
        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
                f"panda_joint{i}"
            ).qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (
                J[:, dofadr].T @ np.diag(self.K) @ error
            )
            self.data.actuator(f"panda_joint{i}").ctrl -= (
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )

    def get_target(self):
        current_position = self.data.body("panda_link7").xpos.copy()
        current_orientation = self.data.body("panda_link7").xquat.copy()
        
        pos_delta = [random.uniform(-self.max_pos_delta, self.max_pos_delta) for _ in range(3)]
        ori_delta = [random.uniform(-self.max_ori_delta, self.max_ori_delta) for _ in range(4)]
        pos_delta = [0,0,-0.2]
        ori_delta = [0,0,0,0]

        target_position = current_position + pos_delta
        target_orientation = current_orientation + ori_delta
        return target_position, target_orientation
    
    def simulate(self):
        action_sampler = 1
        target_pos, target_ori = self.get_target()
        while True:
            if sim.viewer.is_alive:
                action_sampler +=1
                if action_sampler % 30 == 0:
                    action_sampler = 1
                    target_pos, target_ori = self.get_target()
                self.control(target_pos,target_ori)
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()
                #time.sleep(1.0 / self.fps)
            else:
                break

if __name__ == "__main__":
    sim = Sim()
    sim.set_pose(sim.initial_pose)
    sim.simulate()


