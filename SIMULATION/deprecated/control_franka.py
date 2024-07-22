import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from pynput import keyboard
from mujoco_lm_nullspace_ik import IK_Solver
from transformations import euler_from_quaternion

def on_press(key):
    global goal_pos, goal_quat, rot
    delta = 0.02
    if rot == False:
        try:
            if key == keyboard.Key.cmd_r:
                rot = True
                return
            elif key == keyboard.Key.space:
                goal_pos[2] += delta  # Move up
            elif key == keyboard.Key.cmd:
                goal_pos[2] -= delta  # Move down
            elif key == keyboard.Key.left:
                goal_pos[1] -= delta  # Move left
            elif key == keyboard.Key.right:
                goal_pos[1] += delta  # Move right
            elif key == keyboard.Key.up:
                goal_pos[0] -= delta  # Move forward
            elif key == keyboard.Key.down:
                goal_pos[0] += delta  # Move back
        except AttributeError:
            pass
    else:
        euler_angles = [*euler_from_quaternion(goal_quat)]
        try:
            if key == keyboard.Key.cmd_r:
                rot = False
                return
            elif key == keyboard.Key.space:
                euler_angles[2] += delta  # Move up
            elif key == keyboard.Key.cmd:
                euler_angles[2] -= delta  # Move down
            elif key == keyboard.Key.left:
                euler_angles[1] -= delta  # Move left
            elif key == keyboard.Key.right:
                euler_angles[1] += delta  # Move right
            elif key == keyboard.Key.up:
                euler_angles[0] -= delta  # Move forward
            elif key == keyboard.Key.down:
                euler_angles[0] += delta  # Move back

            mujoco.mju_euler2Quat(goal_quat, euler_angles, 'XYZ')

        except AttributeError:
            pass

def main() -> None:
    global goal_pos, goal_quat, rot
    rot = False
    goal_quat = np.zeros(4)
    goal_pos = np.zeros(3)

    # Load the model and data.
    script_dir = Path(__file__).resolve().parent
    xml_path = str(script_dir / "franka_emika_panda" / "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    dt = 0.02
    model.opt.timestep = dt
    # End-effector site we wish to control
    site_name = "attachment_site"
    site_id = model.site("attachment_site").id
    # Enable gravity compensation
    model.body_gravcomp[:] = 1.0
    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    home_key_name = "home"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        key_id = model.key(home_key_name).id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_step(model, data)
        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        goal_pos = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(goal_quat, data.site(site_id).xmat)

        # Start keyboard listener
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        Knull = np.asarray([0.1]*7)
        q0 = data.qpos.copy()
        ik = IK_Solver(model, site_name, Knull, q0)

        while viewer.is_running():

            step_start = time.time()

            # Step the simulation.
            q = ik.solve(data,goal_pos, goal_quat)
            data.ctrl[actuator_ids] = q[dof_ids]
            #data.qpos = q
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    listener.stop()

if __name__ == "__main__":
    main()