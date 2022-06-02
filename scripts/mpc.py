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

    def action(self, obs):
        X = cp.Variable((self.nx, self.T))
        U = cp.Variable((self.nu, self.T))

        cost = 0.0
        constraints = []

        x0      = obs[0:2]
        ego_vel = obs[2]
        nd = obs[3]
        C  = np.array([[0, -nd],
                       [0, 0]])
        xmin = [5.5, 0]

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
                scaling = np.array([max(x0[0],self.SAFETY_DIST), self.VELOCITY_LIMIT])
                # cost += cp.quad_form((X[:, t] - (xmin + C @ X[:, t]))/scaling, self.Q)
                xref = [xmin[0] + ego_vel*nd, 0]
                cost += cp.quad_form((X[:, t] - xref)/scaling, self.Q)

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
    

# Ts = 1

# A = np.array([[1, Ts],
#               [0,  1]])

# B = np.array([[-1/2*Ts*Ts, 0],
#               [-Ts, 0]])
# B = np.array([[-1/2*Ts*Ts],
#               [-Ts]])

# T = 100
# t = np.arange(T)

# Q = np.diag([10, 10])
# R = np.array([5])
# # R = np.diag([5, 5])

# modelMPC = MPCLinear(A, B, Q, R, T)

# x0 = [145, 0]
# xref = [5, 0]

# u, y = modelMPC.action(x0, xref)

# # Plot the results
# plt.subplot(3, 1, 1)
# plt.plot(t, y[0])
# plt.plot([0,T], [5,5])
# plt.xlabel("t [sec]")
# plt.ylabel("s [m]")

# plt.subplot(3, 1, 2)
# plt.plot(t, y[1])
# plt.xlabel("t [sec]")
# plt.ylabel("v [m/s]")

# plt.subplot(3, 1, 3)
# plt.plot(t, u[0])
# plt.xlabel("t [sec]")
# plt.ylabel("u")

# plt.show()