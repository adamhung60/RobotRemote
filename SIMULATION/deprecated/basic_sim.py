import time
from threading import Thread
import glfw
import mujoco
import numpy as np
from pathlib import Path
import mujoco

class Sim:
    
    def __init__(self, fps = 30, start_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
        # initialize model

        self.start_pose = start_pose
        self.fps = fps  # Rendering framerate.
        self.impedances = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
        self.height, self.width = 480, 640  # Rendering window resolution.

        script_dir = Path(__file__).resolve().parent
        xml_path = str(script_dir / "panda_mujoco" / "world.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # setting up camera
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        #setting initial pose
        for i in range(1, 8): 
            self.data.joint(f"panda_joint{i}").qpos = self.start_pose[i - 1]
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        while self.run:
            mujoco.mj_step(self.model, self.data)
            time.sleep(1e-3)

    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Sim", None, None)
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