import rockit
from rockit.sampling_method import SamplingMethod
import random
import string
import numpy as np
import casadi as cs
# class ProblemDimensions:
#     def __init__(self):
#         self.K = 0
#         self.nx = 0
#         self.nu = 0
#         self.ngI = 0
#         self.ng = 0
#         self.ngF = 0
#         self.ngineqI = 0
#         self.ngineq = 0
#         self.ngineqF = 0
#         self.name = "" 
def SimulateOneStep(stage:rockit.Stage, x0, u0, M, dt, intg):
    res = stage.x
    for i in range(len(x0)):
        res = cs.substitute(res, x0[i][0], x0[i][1])
    x0 = cs.evalf(res)
    X_res = np.zeros(stage.nx)
    intg = SamplingMethod(M=M, intg = intg).discrete_system(stage)
    # ret = Function('F', [X0, U, T, t0, P], [X[-1], hcat(X), hcat(poly_coeffs), quad, Zs[-1], hcat(Zs), hcat(poly_coeffs_z)],
    #                ['x0', 'u', 'T', 't0', 'p'], ['xf', 'Xi', 'poly_coeff', 'qf', 'zf', 'Zi', 'poly_coeff_z'])
    X_res[:] = cs.evalf(intg(x0, u0, dt, 0, np.array([]))[0]).T
    return X_res


class SolverStats:
    def __init__(self):
        self.no_iterations = 0
        self.time_FE = 0.
        self.time_solver = 0. # without FE
        self.problem_name = ""
        self.solver = ""
    def __str__(self) -> str:
        return  str(self.problem_name) + " solved with " + str(self.solver) + \
        "\nno_iterations: " + str(self.no_iterations) + \
        "\ntime_FE:       " + str(self.time_FE) + \
        "\ntime_solver:   " + str(self.time_solver) + \
        "\ntime_total:   " + str(self.time_solver + self.time_FE) 
def SolveFatrop(prob, N, name = None):
    print("###############################################")
    print("Solving with fatrop")
    print("###############################################")
    # generate a string of 5 random characters
    if name is None:
        name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    name_f = "benchmark/" + name
    method = rockit.external_method('fatrop', N=N)
    method.set_name(name_f)
    prob.ocp.method(method)
    prob.ocp.solve()
    res = prob.ocp.solve()
    try:
        fatropstats = prob.ocp._method.myOCP.get_stats()
    except:
        fatropstats = prob.ocp._method.myOCP.GetStats()
    stats = SolverStats()
    stats.solver = "fatrop"
    stats.problem_name = name
    stats.no_iterations = fatropstats.iterations_count
    stats.time_FE = fatropstats.eval_hess_time + fatropstats.eval_jac_time + fatropstats.eval_cv_time + fatropstats.eval_grad_time + fatropstats.eval_obj_time
    stats.time_solver = fatropstats.time_total - stats.time_FE
    return res, prob, stats
def SolveIpopt(prob, N, name = None, jit = True):
    print("###############################################")
    print("Solving with ipopt")
    print("###############################################")
    prob.ocp.method(rockit.MultipleShooting(N=N))
    jit_options = {"flags": ["-Ofast -march=native"], "verbose": True}
    prob.ocp.solver("ipopt", {"expand":True, "jit":jit, "compiler":"shell", "jit_options":jit_options,  "ipopt.tol": 1e-8, "ipopt.gamma_theta":1e-12, "ipopt.mu_init":1e2, "ipopt.linear_solver":"ma57", "ipopt.kappa_d":1e-5, "ipopt.print_level":5, "ipopt.min_refinement_steps":0, "ipopt.residual_ratio_max":1e-6})
    # prob.ocp.solver("ipopt", {"expand":True, "ipopt.tol": 1e-8, "ipopt.gamma_theta":1e-12, "ipopt.mu_init":1e2, "ipopt.linear_solver":"ma57", "ipopt.kappa_d":1e-5, "ipopt.print_level":5, "ipopt.print_timing_statistics":"yes"})
    prob.ocp.solve()
    res = prob.ocp.solve()
    ipoptstats = res.stats
    stats = SolverStats()
    stats.solver = "ipopt"
    stats.problem_name = name
    stats.no_iterations = ipoptstats["iter_count"]
    stats.time_FE = ipoptstats["t_wall_nlp_f"] + ipoptstats["t_wall_nlp_g"] + ipoptstats["t_wall_nlp_grad_f"] + ipoptstats["t_wall_nlp_jac_g"]+ ipoptstats["t_wall_nlp_grad"] + ipoptstats["t_wall_nlp_hess_l"]
    stats.time_solver = ipoptstats["t_wall_total"] - stats.time_FE
    return res, prob, stats
def SolveAcados(prob, N, mode = "SQP", qp_solver_cond = 1, name = None, GAUSS_NEWTON_RTI = True):
    print("###############################################")
    print("Solving with Acados")
    print("###############################################")
    # generate a string of 5 random characters
    if name is None:
        name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    if mode == "SQP_RTI":
        method = rockit.external_method('acados', expand = True, N=N,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='GAUSS_NEWTON' if GAUSS_NEWTON_RTI else "EXACT",regularize_method = 'NO_REGULARIZE' if GAUSS_NEWTON_RTI else "CONVEXIFY",integrator_type='ERK',nlp_solver_type='SQP_RTI',qp_solver_cond_N=N//qp_solver_cond, qp_solver_ric_alg = 1, ext_fun_compile_flags = '-Ofast -march=native', hpipm_mode = 'BALANCE')
    if mode == "SQP":
        method = rockit.external_method('acados', expand = True, N=N,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=N//qp_solver_cond, qp_solver_ric_alg = 1, ext_fun_compile_flags = '-Ofast -march=native', hpipm_mode = 'BALANCE')
    prob.ocp.expand = True
    prob.ocp.method(method)
    res = prob.ocp.solve()
    # fatropstats = prob.ocp._method.myOCP.GetStats()
    stats = SolverStats()
    stats.solver = "adados_" + str(mode)
    stats.problem_name = name + "_" + mode + "_" + str(qp_solver_cond)
    stats.no_iterations = prob.ocp._method.ocp_solver.get_stats("sqp_iter")
    stats.time_FE = prob.ocp._method.ocp_solver.get_stats("time_lin") + prob.ocp._method.ocp_solver.get_stats("time_sim")     # stats.time_solver = fatropstats.time_total - stats.time_FE
    # stats.time_FE = 0.0
    stats.time_solver = prob.ocp._method.ocp_solver.get_stats("time_tot") - stats.time_FE
    return res, prob, stats