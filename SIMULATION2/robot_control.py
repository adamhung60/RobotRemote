import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from pynput import keyboard
from pinocchio_ik_solver import IK_Solver, Robot_Workspace
from transformations import euler_from_quaternion

class Simulation():

    def __init__(self, xml_path, urdf_path, workspace, end_effector_joint_index, dt = 0.02, grav_comp = True, site_name = "attachment_site", home_key_name = "home"):
        self.urdf_path = urdf_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.zeros(4)
        self.rot = False
        self.end_effector_joint_index = end_effector_joint_index

        self.dt = dt
        self.model.opt.timestep = dt

        self.site_name = site_name
        self.site_id = self.model.site(site_name).id

        self.model.body_gravcomp[:] = float(grav_comp)

        #self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        #self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])
    
        self.home_key_id = self.model.key(home_key_name).id

        self.workspace = workspace
    
    def simulate(self):

        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
            ) as viewer:

            # Reset the simulation to the initial keyframe.
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
            mujoco.mj_step(self.model, self.data)
            # Initialize the camera view to that of the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

            # Toggle site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            self.goal_pos = self.data.site(self.site_id).xpos.copy()
            mujoco.mju_mat2Quat(self.goal_quat, self.data.site(self.site_id).xmat)

            q0 = np.array(self.data.qpos.copy())
            ik = IK_Solver(self.urdf_path, self.workspace, q0, self.end_effector_joint_index)

            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()
            while viewer.is_running():

                step_start = time.time()
                mat_goal = np.zeros(9)
                #print(mat_goal)
                #print(self.goal_quat)
                mujoco.mju_quat2Mat(mat_goal, self.goal_quat)
                mat_goal = mat_goal.reshape((3,3))
                q = ik.solve(self.goal_pos, mat_goal)
                #self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                self.data.qpos = q
                mujoco.mj_step(self.model, self.data)

                viewer.sync()
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            listener.stop()

    def on_press(self, key):
        delta = 0.01
        if self.rot == False:
            try:
                if key == keyboard.Key.cmd_r:
                    self.rot = True
                    return
                elif key == keyboard.Key.space:
                    self.goal_pos[2] += delta  # Move up
                elif key == keyboard.Key.shift_l:
                    self.goal_pos[2] -= delta  # Move down
                elif key == keyboard.Key.left:
                    self.goal_pos[1] -= delta  # Move left
                elif key == keyboard.Key.right:
                    self.goal_pos[1] += delta  # Move right
                elif key == keyboard.Key.up:
                    self.goal_pos[0] -= delta  # Move forward
                elif key == keyboard.Key.down:
                    self.goal_pos[0] += delta  # Move back
            except AttributeError:
                pass
        else:
            euler_angles = [*euler_from_quaternion(self.goal_quat)]
            try:
                if key == keyboard.Key.cmd_r:
                    self.rot = False
                    return
                elif key == keyboard.Key.space:
                    euler_angles[2] += delta  # Move up
                elif key == keyboard.Key.shift_l:
                    euler_angles[2] -= delta  # Move down
                elif key == keyboard.Key.left:
                    euler_angles[1] -= delta  # Move left
                elif key == keyboard.Key.right:
                    euler_angles[1] += delta  # Move right
                elif key == keyboard.Key.up:
                    euler_angles[0] -= delta  # Move forward
                elif key == keyboard.Key.down:
                    euler_angles[0] += delta  # Move back

                mujoco.mju_euler2Quat(self.goal_quat, euler_angles, 'XYZ')

            except AttributeError:
                pass

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    robot = "panda"

    if robot == "panda":
        xml_path = str(script_dir / "franka_emika_panda" / "scene.xml")
        urdf_path = str(script_dir / "panda.urdf")
        workspace = Robot_Workspace("sphere", center = [0,0,(-.360+1.190)/2], radius = 0.855 - 0.1, height = (-.360+1.190)/2)
        end_effector_joint_index = 7

    elif robot == "lite6":
        xml_path = str(script_dir / "ufactory_lite6" / "scene.xml")
        urdf_path = str(script_dir / "lite6.urdf")
        workspace = Robot_Workspace("sphere", center = [0,0,(-.165+.6835)/2], radius = .440 - 0.1, height = (-.165+.6835)/2)
        end_effector_joint_index = 6

    sim = Simulation(xml_path, urdf_path, workspace, end_effector_joint_index)
    sim.simulate()