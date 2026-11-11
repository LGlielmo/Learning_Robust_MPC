"""Quick speed test for the LMPC unicycle example (same code as the notebook).

Run from Terminal, inside the Ex2 folder:
    conda activate lmpc
    python timing_test.py

It prints which Python/CasADi is used, then runs the first LMPC setting
(l=1, P=15) for 2 iterations and reports the time per LMPC step,
split into "build NLP" (CasADi + IPOPT setup) and "solve" (IPOPT).
For reference: Ugo's ~5 min total corresponds to roughly 0.05-0.1 s per step.
"""
import os, sys, time, platform
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.append(os.path.join(here, '..', '..', 'utils'))

import numpy as np
import casadi
from fnc.LMPC_CT import LMPC_CT
from fnc.MIX_OL_CL import MIX_OL_CL
from simulators import SIMULATOR

print("python :", sys.executable)
print("machine:", platform.machine(), "| python", platform.python_version())
print("numpy", np.__version__, "| casadi", casadi.__version__, "| casadi from:", os.path.dirname(casadi.__file__))
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    print(f"{v} = {os.environ.get(v)}")
print()

radius, dt, N, rhw = 10.0, 0.5, 4, 2.0
simulator = SIMULATOR("unicycle", radius=radius, dt=dt)
mix = MIX_OL_CL(radius)
x0 = [0, 0, 0]
xf, uf = [x0], []
T = 101
for t in range(T - 1):
    mix.solve(xf[t], t, T)
    uf.append(mix.uPred)
    xf.append(simulator.sim(xf[t], mix.uPred))

lmpc = LMPC_CT(N, dt, rhw, l=1, P=15, verbose=False)
s0 = xf[-1][0]
Xf = np.array([[s0, -rhw, 0.0], [s0 + 0.5, rhw, 0.0]]).T
lmpc.set_xf(Xf)
Xv = np.concatenate((Xf, Xf), axis=1); Xv[1, 2] = rhw; Xv[1, 3] = -rhw
lmpc.Xf_vertices = Xv
lmpc.addTrajectory(xf, uf)

# wrap the two expensive methods with timers
t_build, t_solve = [], []
_build, _solve = lmpc.buildNonlinearProgram, lmpc.solve_ftocp
def build(*a, **k):
    t = time.perf_counter(); r = _build(*a, **k); t_build.append(time.perf_counter() - t); return r
def solve(*a, **k):
    t = time.perf_counter(); r = _solve(*a, **k); t_solve.append(time.perf_counter() - t); return r
lmpc.buildNonlinearProgram, lmpc.solve_ftocp = build, solve

t0 = time.perf_counter()
for it in range(2):
    xcl, ucl = [x0], []
    while not lmpc.checkTaskCompletion(xcl[-1]):
        lmpc.solve(xcl[-1])
        u = lmpc.uPred[:, 0]
        ucl.append(u)
        xcl.append(simulator.sim(xcl[-1], u))
    lmpc.addTrajectory(xcl, ucl)
    print(f"iteration {it+1}: {len(xcl)-1} steps (Ugo's notebook: {[84, 69][it]})")
tot = time.perf_counter() - t0
n = len(t_build)
print(f"\nLMPC steps: {n}   total: {tot:.1f} s   per step: {tot/n:.3f} s")
print(f"  build NLP: {np.mean(t_build):.3f} s/step   solve (4 IPOPT calls): {np.mean(t_solve):.3f} s/step")
