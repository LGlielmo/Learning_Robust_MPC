# =====================================================================================
# LMPC_CT.py  --  COMMENTED COPY (the code is identical to Ugo's fnc/LMPC_CT.py;
#                 only comments were added and tabs were turned into spaces).
#
# "CT" = Convex, Time-varying safe set.
#
# Notation used in the comments
#   n = 3 states   x = (s, e, v): progress along the road, lateral offset, speed
#   d = 2 inputs   u = (theta, a): heading angle in the X-Y frame, acceleration
#   N             prediction horizon
#   SS            "sampled safe set": all stored closed-loop trajectories
#   Vfun          cost-to-go of every stored state (steps still needed to reach the goal)
#   lambda        weights of the convex combination of safe-set points
#
# Python reminders
#   class X(object):  defines a new type of object; "def" inside it defines its methods
#   self.something    a variable stored inside the object (it survives between calls)
#   np.concatenate((A,B), axis=1)  = [A, B] in MATLAB ; axis=0 = [A; B]
#   A[:, idx]         columns idx of A ;  A.T  transpose ;  A.shape[1]  number of columns
#   np.arange(a, b)   = a, a+1, ..., b-1  (b excluded)
# =====================================================================================

import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools
from casadi import *        # CasADi: symbolic variables (SX), constraints (vertcat, mtimes) and the call to IPOPT (nlpsol)
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *         # cvxpy is imported but not used in this file
# NOTE: the three "import *" lines dump all names of casadi, numpy and cvxpy in the same space,
# so later imports can hide earlier names. It works with the versions Ugo used; if you ever get a
# strange error in this file with newer library versions, this is the first suspect.


class LMPC_CT(object):
    """Learning Model Predictive Controller (LMPC) implemented using the Convex Time varing safe set (CT)
        Inputs:
            - ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
            - l: number of past trajectories used to construct the local safe set and local Q-function
            - M: number of data points from each trajectory used to construct the local safe set and local Q-function
        Methods:
            - addTrajectory: adds a trajectory to the safe set SS and update value function
            - computeCost: computes the cost associated with a feasible trajectory
            - solve: uses ftocp and the stored data to comptute the predicted trajectory
            - closeToSS: computes the K-nearest neighbors to zt"""
    # (the docstring above is partly outdated: "M" is called P in the code, and there is no
    #  ftocp, computeCost or closeToSS method in this version)

    # ---------------------------------------------------------------------------------
    # __init__ is the constructor: it runs once when the notebook writes
    #     lmpc = LMPC_CT(N, dt, road_half_width, l=l, P=P, verbose=False)
    # and creates the (still empty) memory of the controller.
    # ---------------------------------------------------------------------------------
    def __init__(self, N, dt, roadHalfWidth, l, P, Xf_vertices = None, verbose = False):
        self.N = N                  # horizon
        self.n = 3                  # number of states
        self.d = 2                  # number of inputs
        self.radius = 10.0          # road radius R (hard-coded here: must match the notebook)
        self.optCost= np.inf        # optimal cost of the last optimisation (inf = nothing solved yet)
        self.dt = dt                # sampling time
        self.dimSS = []             # number of points in the local safe set (set later)
        self.roadHalfWidth = roadHalfWidth

        self.SS    = []             # list of stored state trajectories, one 3 x (T+1) array per iteration
        self.uSS   = []             # list of stored input trajectories, one 2 x (T+1) array per iteration
        self.Vfun  = []             # list of cost-to-go vectors, one per iteration
        self.l     = l              # how many past iterations are used
        self.P     = P              # how many points per iteration are used
        self.zt    = []
        self.it    = 0              # number of stored iterations
        self.timeVarying = True
        self.itCost = []            # time to complete the task at each stored iteration
        self.verbose = verbose
        self.Xf_vertices = Xf_vertices   # corners of the goal box (set by the notebook)

    # ---------------------------------------------------------------------------------
    # addTrajectory: called after each completed iteration (and once with the feasible
    # trajectory). This is where the controller "learns": it stores the data.
    #   x : list of states  [x_0, ..., x_T]
    #   u : list of inputs  [u_0, ..., u_{T-1}]
    # ---------------------------------------------------------------------------------
    def addTrajectory(self, x, u):
        # Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(np.array(x).T)       # store the states as a 3 x (T+1) matrix (one state per column)
        # store the inputs as a 2 x (T+1) matrix: a zero input is appended for the final state,
        # because once in the goal (stopped) the car stays there with u = 0
        self.uSS.append(np.concatenate( (np.array(u).T, np.zeros((np.array(u).T.shape[0],1)) ), axis=1)) # Here concatenating zero as f(xf, 0) = xf by assumption

        # Compute and store the cost associated with the feasible trajectory
        # cost-to-go of each stored state = steps still needed: [T, T-1, ..., 1, 0]
        self.Vfun.append(np.arange(np.array(x).T.shape[1]-1,-1,-1))

        # Compute initial guess for nonlinear solver and store few variables
        # warm start for the first optimisation of the next iteration: first N+1 states and first N
        # inputs of this trajectory, stacked into one long vector (the order used by the solver)
        self.xGuess = np.concatenate((np.array(x).T[:,0:(self.N+1)].T.flatten(), np.array(u).T[:,0:(self.N)].T.flatten()), axis = 0)

        # Initialize cost varaibles for bookkeeping
        self.cost    = self.Vfun[-1][0]     # cost-to-go of x_0 = T = time to complete the task in this iteration
        self.itCost.append(self.cost)
        self.optCost = self.cost + 1
        self.oldIt  = self.it

        # Pass inital guess to ftopc object
        self.xSol = np.array(x).T[:,0:(self.N+1)]
        self.uSol = np.array(u).T.reshape(self.d,-1)[:,0:(self.N)]

        # Update time Improvement counter
        self.timeImprovement = 0

        # Update iteration counter
        self.it = self.it + 1

        # Update indices of stored data points used to contruct the local safe set and Q-function
        # For every stored iteration i (length Tj) choose the P consecutive time indices that will be
        # used at time t = 0 of the next iteration:  Tj - T* + N, ..., Tj - T* + N + P - 1,
        # where T* is the best time so far. Meaning: align all trajectories "from the end" with
        # the fastest one, and look N steps ahead. These indices move forward by one at every time
        # step (see timeSS), which makes the safe set TIME-VARYING.
        self.SSindices =[]
        Tstar = np.min(self.itCost)
        for i in range(0, self.it):
            Tj = np.shape(self.SS[i])[1]-1
            self.SSindices.append(np.arange(Tj - Tstar + self.N, Tj - Tstar + self.N+self.P))

    # ---------------------------------------------------------------------------------
    # solve: called by the notebook at every time step with the measured state xt.
    # Result: self.uPred (2 x N predicted inputs); the notebook applies only uPred[:,0].
    # ---------------------------------------------------------------------------------
    def solve(self, xt, verbose = False):

        # First retive the data points used to cconstruct the safe set.
        minIt = np.max([0, self.it - self.l])   # use only the last l iterations
        SSVfun = []     # for each used iteration: chosen states with their cost-to-go (rows: s, e, v, V)
        SSnext = []     # for each chosen state: its SUCCESSOR state and input in the stored data (for the warm start)
        # Loop over j-l iterations used to contruct the local safe set
        for i in range(minIt, self.it):
            # idx associated with the data points from iteration i which are in the local safe set
            idx = self.timeSS(i)
            # Stored state and cost value (Check if using terminal state or terminal point)
            if (self.Xf_vertices is not None) and (idx[-1] == np.shape(self.SS[i])[1]-1):
                # The window has reached the end of trajectory i: also add the 4 corners of the goal
                # box with cost-to-go 0, so the prediction can end anywhere in the goal box,
                # not only at the stored final point.
                augSS   = np.concatenate((self.SS[i][:,idx], self.Xf_vertices), axis=1 )
                augCost = np.concatenate((self.Vfun[i][idx], np.zeros(self.Xf_vertices.shape[1])), axis=0 )
                SSVfun.append( np.concatenate( (augSS, [augCost]), axis=0 ).T )

                # Store the successors of the states into the safe set and the control action. (Note that the vertices of X_f are invariant states)
                # This matrix will be used to compute the vector zt which represent a feasible guess for the ftocp at time t+1
                xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
                extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)   # repeat the last column (final state is invariant)
                verticesAndInputs = np.concatenate((self.Xf_vertices, np.zeros((self.d, self.Xf_vertices.shape[1]))), axis=0)
                SSnext.append(np.concatenate((extendedSS[:,idx+1], verticesAndInputs), axis = 1).T)
            else:
                # Normal case: just the P chosen stored states and their cost-to-go
                SSVfun.append( np.concatenate( (self.SS[i][:,idx], [self.Vfun[i][idx]]), axis=0 ).T )
                # Store the successors of the states into the safe set and the control action.
                # This matrix will be used to compute a feasible guess for the ftocp at time t+1
                xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
                extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)
                SSnext.append(extendedSS[:,idx+1].T)


        # From a 3D list to a 2D array
        # Stack the data of all used iterations into one matrix with one column per safe-set point:
        #   SSVfun_vector  rows 0..2 = state, row 3 = cost-to-go
        #   SSnext_vector  rows 0..2 = successor state, rows 3..4 = input applied there
        SSVfun_vector = np.squeeze(list(itertools.chain.from_iterable(SSVfun))).T
        SSnext_vector = np.squeeze(list(itertools.chain.from_iterable(SSnext))).T

        # Add dimension if needed
        if SSVfun_vector.ndim == 1:
            SSVfun_vector = np.array([SSVfun_vector]).T

        # Now update ftocp with local safe set
        self.buildNonlinearProgram( SSVfun_vector)   # build the optimisation problem for THIS safe set

        # Now solve ftocp
        self.solve_ftocp(xt)                         # solve it from the current state

        # Assign input
        self.uPred = self.uSol

        # Update guess for the ftocp using optimal predicted trajectory and multipliers lambda
        # Warm start for the next time step (standard LMPC "shifted" candidate):
        #   drop the first predicted state/input, and append, as new last state and input, the same
        #   convex combination (weights lambda) of the SUCCESSORS of the safe-set points.
        if self.optCost > 1:
            xfufNext  = np.dot(SSnext_vector, self.lamb)
            # Update initial guess
            xflatOpenLoop  = np.concatenate( (self.xSol[:,1:(self.N+1)].T.flatten(), xfufNext[0:self.n,0]), axis = 0)
            uflatOpenLoop  = np.concatenate( (self.uSol[:,1:(self.N)].T.flatten()  , xfufNext[self.n:(self.n+self.d),0]), axis = 0)
            self.xGuess = np.concatenate((xflatOpenLoop, uflatOpenLoop) , axis = 0)

    # ---------------------------------------------------------------------------------
    # timeSS: returns the time indices of iteration "it" to use now, then moves the window
    # one step forward for the next call (time-varying safe set).
    # ---------------------------------------------------------------------------------
    def timeSS(self, it):
        # This function computes the indices used to construct the safe set
        # self.SSindices[it] is initialized when the trajectory is added to the safe set after computing \delta^i and P

        # Read the time indices
        currIdx = self.SSindices[it]
        # By definition we have x_t^j = x_F \forall t > T^j ---> check indices to select
        # currIdxShort = currIdx[ (currIdx >0) & (currIdx < np.shape(self.SS[it])[1])]
        currIdxShort = currIdx[ currIdx < np.shape(self.SS[it])[1] ]   # drop indices beyond the end of the trajectory

        # Progress time indices
        self.SSindices[it] = self.SSindices[it] + 1                    # next time step: window shifted by one

        # If there is just one time index --> add dimension
        if np.shape(currIdxShort)[0] < 1:
            currIdxShort = np.array([np.shape(self.SS[it])[1]-1])       # if nothing is left, use the final state

        return currIdxShort


    # ---------------------------------------------------------------------------------
    # set_xf: store the goal set. xf has 2 columns (lower and upper bounds) for a box,
    # or 1 column for a single goal point.
    # ---------------------------------------------------------------------------------
    def set_xf(self, xf):
            # Set terminal state
            if xf.shape[1] >1:
                self.terminalSet = True
                self.xf_lb = xf[:,0]
                self.xf_ub = xf[:,1]
            else:
                self.terminalSet = False
                self.xf    = xf[:,0]
                self.xf_lb = self.xf
                self.xf_ub = self.xf

    # ---------------------------------------------------------------------------------
    # checkTaskCompletion: True if the state x is inside the goal box
    # (or, for a goal point, closer than 1e-4 to it). Used by the notebook's time loop.
    # ---------------------------------------------------------------------------------
    def checkTaskCompletion(self,x):
        # Check if the task was completed
        taskCompletion = False
        if (self.terminalSet == True) and (self.xf_lb <= x).all() and (x <= self.xf_ub).all():
            taskCompletion = True
        elif (self.terminalSet == False) and np.dot(x-self.xf, x-self.xf)<= 1e-8:
            taskCompletion = True

        return taskCompletion


    # ---------------------------------------------------------------------------------
    # solve_ftocp: solve the finite-time optimal control problem from x0.
    #
    # The stage cost is 1 outside the goal and 0 inside: not differentiable, so IPOPT cannot
    # handle it directly. Trick: try every possible case. For i = 0, 1, ..., N-1 force the LAST i
    # predicted states into the goal box; then the stage cost is simply known (number of steps
    # outside the goal) and only the terminal cost is optimised. Case i = N (reach the goal in one
    # step) is only checked on the warm start. The best of all cases wins.
    #
    # Order of the decision vector (one long vector for IPOPT):
    #   [ x_0..x_N (3(N+1)) | u_0..u_{N-1} (2N) | lambda (dimSS) | slack (3) ]
    # ---------------------------------------------------------------------------------
    def solve_ftocp(self, x0, verbose=False):
        # Initialize initial guess for lambda
        lambGuess = np.hstack( (np.ones(self.dimSS)/self.dimSS, np.zeros(self.n)) )   # lambda uniform, slack 0
        lambGuess[0] = 1
        self.xGuessTot = np.hstack( (self.xGuess, lambGuess) )      # complete initial guess for IPOPT

        # lambGuess = np.concatenate((np.ones(self.dimSS)/self.dimSS, np.zeros(self.n)), axis = 0)
        # lambGuess[0] = 1
        # self.xGuessTot = np.concatenate( (self.xGuess, lambGuess), axis=0 )

        # Need to solve N+1 ftocp as the stage cost is the indicator function --> try all configuration
        costSolved = []     # cost of each case (inf if infeasible)
        soluSolved = []     # solution of each case
        slackNorm  = []
        for i in range(0, self.N+1):
            # IMPORTANT: here 'i' represents the number of states constrained to the safe set --> the horizon length is (N-i)
            if i is not self.N:     # (Python note: "is not" compares identity; "!=" would be the clean way, it works here for small integers)
                # Set box constraints on states (here we constraint the last i steps of the horizon to be xf)
                # lbx/ubx = lower/upper bounds for every entry of the decision vector:
                #   x_0 fixed to the measured state (lb = ub = x0)
                #   N-i free states:  s in [-100,100], e in [-2,2] (road), v in [0,500]
                #   last i states:    inside the goal box
                #   N inputs:         theta in [-2,2] rad, a in [-1,1]
                #   lambda >= 0 (upper bound 10 never active since they sum to 1);  slack in [-10,10]
                self.lbx = x0 + [-100, -self.roadHalfWidth, -0]*(self.N-i)+ self.xf_lb.tolist()*i + [-2.0,-1.0]*self.N + [0]*self.dimSS  + [-10]*self.n
                self.ubx = x0 +  [100,  self.roadHalfWidth,  500]*(self.N-i)+ self.xf_ub.tolist()*i + [2.0, 1.0]*self.N + [10]*self.dimSS + [10]*self.n
                # (in Python, list + list concatenates, and [a,b]*k repeats the list k times)

                # Solve nonlinear programm
                sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot.tolist())

                # Check if the solution is feasible
                # (IPOPT succeeded AND the slack is zero, i.e. x_N is really in the convex hull of the safe set)
                idxSlack = (self.N+1)*self.n + self.d*self.N + self.dimSS
                self.slack    = sol["x"][idxSlack:idxSlack+self.n]
                slackNorm.append(np.linalg.norm(self.slack,2))
                if (self.solver.stats()['success']) and (np.linalg.norm(self.slack,2)< 1e-8):
                    self.feasible = 1
                    # Notice that the cost is given by the cost of the ftocp + the number of steps not constrainted to be xf
                    lamb = sol["x"][((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)]
                    terminalCost = np.dot(self.costSS, np.array(lamb)).item()   # sum_k lambda_k V_k ; .item() turns the 1x1 result into a plain number (needed with numpy >= 1.24; Ugo's original line had no .item())
                    # costSolved.append(terminalCost+(self.N+1-i))
                    if i == 0:
                        costSolved.append(terminalCost+(self.N-i))
                    else:
                        costSolved.append(terminalCost+(self.N-(i-1)))

                    soluSolved.append(sol)
                else:
                    costSolved.append(np.inf)
                    soluSolved.append(sol)
                    self.feasible = 0

            else: # if horizon one time step (because N-i = 0) --> just check feasibility of the initial guess
                uGuess = self.xGuess[(self.n*(self.N+1)):(self.n*(self.N+1)+self.d)]
                xNext  = self.f(x0, uGuess)
                slackNorm.append(0.0)
                if self.checkTaskCompletion(xNext):     # does the first guessed input bring us into the goal in one step?
                    self.feasible = 1
                    costSolved.append(1)
                    sol["x"] = self.xGuessTot
                    soluSolved.append(sol)
                else:
                    costSolved.append(np.inf)
                    soluSolved.append(sol)
                    self.feasible = 0


        # Store optimal solution
        # pick the cheapest case and cut the long solution vector back into states, inputs and lambda
        self.optCost = np.min(costSolved)
        x = np.array(soluSolved[np.argmin(costSolved)]["x"])
        self.xSol = x[0:(self.N+1)*self.n].reshape((self.N+1,self.n)).T                                  # 3 x (N+1) predicted states
        self.uSol = x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.N,self.d)).T   # 2 x N predicted inputs
        self.lamb = x[((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)] # weights lambda
        optSlack = slackNorm[np.argmin(costSolved)]

    # ---------------------------------------------------------------------------------
    # buildNonlinearProgram: writes the optimisation problem symbolically with CasADi and
    # creates an IPOPT solver for it. Called at EVERY time step (the safe set changes), which
    # is the main reason why the notebook is slow.
    #
    #   min    sum_k lambda_k V_k  +  1e12 * ||slack||^2
    #   s.t.   x_{k+1} = f(x_k, u_k),  k = 0..N-1          (unicycle dynamics)
    #          x_N + slack = sum_k lambda_k x_k^SS          (terminal state in the convex hull of the safe set)
    #          sum_k lambda_k = 1,  lambda >= 0             (convex combination; lambda >= 0 is in lbx)
    #          + the bounds lbx/ubx set in solve_ftocp
    # The slack makes the problem always solvable; its huge weight drives it to zero whenever possible.
    # ---------------------------------------------------------------------------------
    def buildNonlinearProgram(self, SSVfun):
        # Define variables
        n = self.n
        d = self.d
        N = self.N
        X      = SX.sym('X', n*(N+1));      # symbolic predicted states, stacked: [s0,e0,v0, s1,e1,v1, ...]
        U      = SX.sym('X', d*N);          # symbolic inputs, stacked: [theta0,a0, theta1,a1, ...]
        dimSS  = np.shape(SSVfun)[1]        # number of safe-set points
        lamb   = SX.sym('X',  dimSS)        # one weight per safe-set point
        xSS    = SSVfun[0:n, :]             # safe-set states (3 x dimSS)
        costSS = np.array([SSVfun[-1, :]])  # their cost-to-go (1 x dimSS)
        slack  = SX.sym('X', n);

        self.dimSS = dimSS
        self.SSVfun = SSVfun
        self.xSS = xSS
        self.costSS = costSS
        self.xSS = xSS
        self.costSS = costSS

        # Define dynamic constraints
        # (same equations as SIMULATOR.unicycle, written as  x_{k+1} - f(x_k,u_k) = 0)
        constraint = []
        for i in range(0, N):
            constraint = vertcat(constraint, X[n*(i+1)+0] - (X[n*i+0] + self.dt*X[n*i+2]*np.cos( U[d*i+0] - X[n*i+0] / self.radius) / (1 - X[n*i+1]/self.radius ) ))
            constraint = vertcat(constraint, X[n*(i+1)+1] - (X[n*i+1] + self.dt*X[n*i+2]*np.sin( U[d*i+0] - X[n*i+0] / self.radius) ))
            constraint = vertcat(constraint, X[n*(i+1)+2] - (X[n*i+2] + self.dt*U[d*i+1]))

        # terminal constraints
        constraint = vertcat(constraint, slack + X[n*N:(n*(N+1))+0] - mtimes( xSS ,lamb) )   # x_N + slack = xSS * lambda
        constraint = vertcat(constraint, 1 - mtimes(np.ones((1, dimSS )), lamb ) )          # sum(lambda) = 1

        # Defining Cost (We will add stage cost later)
        # (the stage cost is added in solve_ftocp as the known number of steps outside the goal)
        cost = mtimes(costSS, lamb) + 1000000**2*(slack[0]**2 + slack[1]**2 + slack[2]**2)
        self.constraint= constraint
        # Set IPOPT options
        # opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        nlp = {'x':vertcat(X,U,lamb,slack), 'f':cost, 'g':constraint}   # decision vector, cost, constraints
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)               # the solver object, called in solve_ftocp

        # Set lower bound of inequality constraint to zero to force: 1) n*N state dynamics, 2) n terminal contraints and 3) CVX hull constraint
        # lbg = ubg = 0  ->  all the constraints above are equalities
        self.lbg_dyanmics = [0]*(n*N) + [0]*(n) + [0]
        self.ubg_dyanmics = [0]*(n*N) + [0]*(n) + [0]



    # ---------------------------------------------------------------------------------
    # f: the model, x_{t+1} = f(x_t, u_t) (numeric version, used for the one-step check above)
    # ---------------------------------------------------------------------------------
    def f(self, x, u):
        # Given a state x and input u it return the successor state
        xNext = np.array([x[0] + self.dt * x[2]*np.cos(u[0] - x[0]/self.radius) / (1 - x[1] / self.radius),
                          x[1] + self.dt * x[2]*np.sin(u[0] - x[0]/self.radius),
                          x[2] + self.dt * u[1]])
        return xNext.tolist()
