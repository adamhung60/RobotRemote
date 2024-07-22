import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from pynput import keyboard
from transformations import euler_from_quaternion

script_dir = Path(__file__).resolve().parent
xml_path = str(script_dir / "franka_emika_panda" / "scene.xml")

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.2

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
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control
    site_id = model.site("attachment_site").id

    # Enable gravity compensation
    model.body_gravcomp[:] = float(gravity_compensation)

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

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_step(model, data)
        goal_pos = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(goal_quat, data.site(site_id).xmat)

        while viewer.is_running():

            step_start = time.time()
        
            # Position error.

            error_pos[:] = goal_pos - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, goal_quat, site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    listener.stop()

if __name__ == "__main__":
    main()