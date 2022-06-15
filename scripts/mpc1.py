from pickle import TRUE
from tabnanny import verbose
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

plt.close('all')

class MPCLinear():
    """
    ### Inputs
        x: State
        u: Control input
        A, B: Defined by x[k+1] = Ax[k] + Bu[k]
        
    """
    def __init__(self, A, B, G, Q, R, T, Ts):
        self.nx = A.shape[1]
        self.nu = B.shape[1]
        
        self.A  = A
        self.B  = B
        self.G  = G 
        self.Q  = Q
        self.R  = R
        self.T  = T

        self.Ts = Ts

        self.fv_vel = None

        self.U = None
        self.X = None

        self.SAFETY_DIST    = 5
        self.THROTTLE_LIMIT = 1
        self.VELOCITY_LIMIT = 40
        self.ACC_LIMIT      = 1.5
        self.JERK_LIMIT     = 2
        self.TH             = 0.0

    def action(self, obs):
        X = cp.Variable((self.nx, self.T))
        U = cp.Variable((self.nu, self.T))

        cost = 0.0
        constraints = []

        x0      = obs[0:5]
        ego_vel = obs[2]
        nd = obs[5]
        C = np.array([[1, 0, -(self.TH + nd), 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        yref = [5.0, 0, 0, 0]

        fv_vel = x0[1] + ego_vel

        if self.fv_vel is None:
            md = 0.0
        else:
            md = (fv_vel - self.fv_vel)/self.Ts # Measured disturbance
        md = np.array([md])
        self.fv_vel = fv_vel

        for t in range(self.T):
            if max(self.R.shape) == 1:
                cost += self.R*cp.square(U[:,t])
            else:
                cost += cp.quad_form(U[:,t], self.R)

            if t != 0:
                scaling = np.array([max(x0[0],self.SAFETY_DIST), self.VELOCITY_LIMIT, self.ACC_LIMIT, self.JERK_LIMIT])
                cost += cp.quad_form((C @ X[:, t] - yref)/scaling, self.Q)

            if t < (self.T - 1):
                constraints += [X[:, t + 1] == self.A @ X[:, t] + self.B @ U[:, t] + self.G @ md]
                if max(self.R.shape) == 1:
                    cost += self.R*cp.square(U[:, t + 1] - U[:, t]) #To prevent chattering
                else: 
                    cost += cp.quad_form(U[:, t + 1] - U[:, t], self.R)

        constraints += [X[:, 0] == x0]
        constraints += [X[0, :] >= self.SAFETY_DIST]
        constraints += [cp.abs(X[1, :]) <= self.VELOCITY_LIMIT]
        constraints += [cp.abs(U[0, :]) <= self.THROTTLE_LIMIT]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        assert prob.is_dpp()

        result = prob.solve(verbose=False, warm_start=True)

        if result == np.inf or result == -np.inf:
            U = self.U[:,1:]
            X = self.X[:,1:]
            self.U = U
            self.X = X
        else:
            self.U = U
            self.X = X

        return U.value, X.value, md

    def action_single(self, x0, xref, ego_vel):
        u, _, _ = self.action(x0, xref, ego_vel)
        try:
            action = np.array([u[0,0]])
        except:
            action = np.array([0]) 
        return action
    

def InitializeMPC(Ts):
    tau = 0.5
    A = np.array([[1, Ts, 0, -1/2*Ts**2, 0],
                  [0, 1, 0, -Ts, 0],
                  [0, 0, 1, Ts, 0],
                  [0, 0, 0, (1-Ts/tau),0],
                  [0, 0, 0, -1/tau, 0]])
    B = 1.5*np.array([[0],
                      [0],
                      [0],
                      [Ts/tau],
                      [1/tau]])
    G = np.array([[1/2*Ts**2],
                  [Ts],
                  [0],
                  [0],
                  [0]])
    T = 20
    Q = np.diag([2, 2, 2, 2])
    R = np.array([10])
    return MPCLinear(A, B, G, Q, R, T, Ts)