import time
from threading import Thread
import glfw
import mujoco
import numpy as np
import random

class Sim:

    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 480, 640  # Rendering window resolution.
    fps = 30  # Rendering framerate.

    def __init__(self) -> None:
        # initialize model
        self.model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
        self.data = mujoco.MjData(self.model)
        # setting up camera
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        # setting initial pose
        for i in range(1, 8): 
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)


    # generates positions and orientations for end effector in real time and passes them to control
    def step(self) -> None:
        xpos0 = self.data.body("panda_link7").xpos.copy()
        xquat0 = self.data.body("panda_link7").xquat.copy()
        max_xdelta = 0.01
        max_qdelta = 0.01
        xbound_radius = 0.01
        xbound_center = xpos0
        while self.run:
            xdelta = [random.uniform(-max_xdelta, max_xdelta) for _ in range(3)]
            qdelta = [random.uniform(-max_qdelta, max_qdelta) for _ in range(4)]
            xpos_d = [max(xbound_center[i]-xbound_radius, 
                        min(xpos0[i] + xdelta[i], xbound_center[i]+xbound_radius)) 
                        for i in range(3)]
            xquat0 = xquat0 # + qdelta
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            time.sleep(1e-3)

    # does position and orientation control
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



    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Sim2", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()


if __name__ == "__main__":
    Sim().start()