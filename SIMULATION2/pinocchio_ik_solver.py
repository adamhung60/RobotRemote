from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
import pinocchio

class IK_Solver():

    def __init__(self, urdf_path, workspace, q0):
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.workspace = workspace
        self.q0 = q0
        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12

        self.JOINT_ID = 7
    
    def solve(self, goal_pos, goal_mat):
        goal_pos = self.workspace.clip_to_workspace(goal_pos)
        goal_ori = goal_mat.reshape((3,3))
        print(goal_ori)
        oMdes = pinocchio.SE3(goal_ori, np.array(goal_pos))

        i = 0
        q = self.q0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.JOINT_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.IT_MAX:
                success = False
                break
            J = pinocchio.computeJointJacobian(self.model, self.data, q, self.JOINT_ID)  # in joint frame
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            if not i % 10:
                print("%d: error = %s" % (i, err.T))
            i += 1

        if success:
            print("Convergence achieved!")
        else:
            print("failed to converge")

        return q.flatten().tolist()
    
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
            
















