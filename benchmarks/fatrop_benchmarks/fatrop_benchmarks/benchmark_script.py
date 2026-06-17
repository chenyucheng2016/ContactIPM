from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados, SolverStats
from cart_pendulum.mpc.problem_specification import CartPendulumMPC
from cart_pendulum.opt_time.problem_specificationT1 import CartPendulumTime
from quadcopter.p2p.problem_specificationT1 import QuadCopterP2P
from quadcopter.mpc.problem_specification import QuadCopterMPC
from quadcopter.one_obstacle.problem_specificationT1 import QuadCopterOneObs
from quadcopter.three_obstacles.problem_specificationT1 import QuadCopterThreeObs
from robot.one_obstacle.problem_specificationT1 import RobotObs
from truck_trailer.motion_planning.problem_specificationT1 import TruckTrailerTime
from hanging_chain.mpc_2D.problem_specification2DMPC import HangingChain2DMPC
from hanging_chain.mpc_3D.problem_specification3DMPC import HangingChain3DMPC
from moonlander.opt_time.problem_specificationT1 import MoonLanderOptTime
import re
def print_header():
    # print("problem name &speedup& no iterations & time solver & time FE \\\\", end="\n")
    print("problem name &  time fatrop & time ipopt & time acados sqp & time acados rti \\\\", end="\n")
# def print_line(stats_fatrop: SolverStats, stats_ipopt: SolverStats):
#     print("{} &".format(re.sub("_", " ", stats_fatrop.problem_name)), end = "")
#     speedup =  (stats_ipopt.time_FE + stats_ipopt.time_solver) / (stats_fatrop.time_FE + stats_fatrop.time_solver)
#     print("{:.2f} &".format(speedup), end = "")
#     print("$\\begin{{array}}{{r}}{} \\\\ {} \\end{{array}}$&".format(stats_fatrop.no_iterations, stats_ipopt.no_iterations), end="")
#     fatrop_time_total = stats_fatrop.time_solver + stats_fatrop.time_FE
#     ipopt_time_total = stats_ipopt.time_solver + stats_ipopt.time_FE
#     print("$\\begin{{array}}{{r}}{:.2e} \\\\ {:.2e} \\end{{array}} $ &".format(fatrop_time_total, ipopt_time_total), end="")
#     print("$\\begin{{array}}{{r}}{:.2e} \\\\ {:.2e} \\end{{array}} $\\\\ \\hline".format(stats_fatrop.time_FE, stats_ipopt.time_FE), end="\n")
# def print_line(stats_fatrop: SolverStats, stats_ipopt: SolverStats):
#     print("{} &".format(re.sub("_", " ", stats_fatrop.problem_name)), end = "")
#     speedup =  (stats_ipopt.time_FE + stats_ipopt.time_solver) / (stats_fatrop.time_FE + stats_fatrop.time_solver)
#     print("{:.2f} &".format(speedup), end = "")
#     print("\\begin{{tabular}}{{r}}{} \\\\ {} \\end{{tabular}}&".format(stats_fatrop.no_iterations, stats_ipopt.no_iterations), end="")
#     fatrop_time_total = stats_fatrop.time_solver + stats_fatrop.time_FE
#     ipopt_time_total = stats_ipopt.time_solver + stats_ipopt.time_FE
#     print("\\begin{{tabular}}{{r}}{:.2f} ms\\\\ {:.2f} ms\\end{{tabular}} &".format(1000*fatrop_time_total, 1000*ipopt_time_total), end="")
#     print("\\begin{{tabular}}{{r}}{:.2f} ms\\\\ {:.2f} ms\\end{{tabular}} \\\\ \\hline".format(1000*stats_fatrop.time_FE, 1000*stats_ipopt.time_FE), end="\n")
def print_line(stats_fatrop: SolverStats, stats_ipopt: SolverStats):
    print("{} &".format(re.sub("_", " ", stats_fatrop.problem_name)), end = "")
    fatrop_time_total = stats_fatrop.time_solver + stats_fatrop.time_FE
    ipopt_time_total = stats_ipopt.time_solver + stats_ipopt.time_FE
    print("{:.2f} ms& {:.2f} ms& & \\\\".format(1000*fatrop_time_total, 1000*ipopt_time_total), end="\n")

if __name__ == "__main__":
    stats_fatrop = []
    stats_ipopt = []
    stats_acados = []
    print("solving CartPendulumMPC with Fatrop")
    stats_fatrop.append(SolveFatrop(CartPendulumMPC(), 25, "cart_pendulum_mpc")[2])
    print("solving CartPendulumMPC with Ipopt")
    stats_ipopt.append(SolveIpopt(CartPendulumMPC(), 25, "cart_pendulum_mpc")[2])
    print("solving CartPendulumTime with Fatrop")
    stats_fatrop.append(SolveFatrop(CartPendulumTime(), 100, "cart_pendulum_time")[2])
    print("solving CartPendulumTime with Ipopt")
    stats_ipopt.append(SolveIpopt(CartPendulumTime(), 100, "cart_pendulum_time")[2])
    print("solving QuadCopterP2P with Fatrop")
    stats_fatrop.append(SolveFatrop(QuadCopterP2P(), 25, "quadcopter_p2p")[2])
    print("solving QuadCopterP2P with Ipopt")
    stats_ipopt.append(SolveIpopt(QuadCopterP2P(), 25, "quadcopter_p2p")[2])
    print("solving QuadCopterMPC with Fatrop")
    stats_fatrop.append(SolveFatrop(QuadCopterMPC(), 25, "quadcopter_mpc")[2])
    print("solving QuadCopterMPC with Ipopt")
    stats_ipopt.append(SolveIpopt(QuadCopterMPC(), 25, "quadcopter_mpc")[2])
    print("solving QuadCopterOneObs with Fatrop")
    stats_fatrop.append(SolveFatrop(QuadCopterOneObs(), 25, "quadcopter_one_obs")[2])
    print("solving QuadCopterOneObs with Ipopt")
    stats_ipopt.append(SolveIpopt(QuadCopterOneObs(), 25, "quadcopter_one_obs")[2])
    print("solving QuadCopterThreeObs with Fatrop")
    stats_fatrop.append(SolveFatrop(QuadCopterThreeObs(), 100, "quadcopter_three_obs")[2])
    print("solving QuadCopterThreeObs with Ipopt")
    stats_ipopt.append(SolveIpopt(QuadCopterThreeObs(), 100, "quadcopter_three_obs")[2])
    print("solving RobotObs with Fatrop")
    stats_fatrop.append(SolveFatrop(RobotObs(), 50, "robot_obs")[2])
    print("solving RobotObs with Ipopt")
    stats_ipopt.append(SolveIpopt(RobotObs(), 50, "robot_obs")[2])
    print("solving TruckTrailerTime with Fatrop")
    stats_fatrop.append(SolveFatrop(TruckTrailerTime(), 50, "truck_trailer_time")[2])
    print("solving TruckTrailerTime with Ipopt")
    stats_ipopt.append(SolveIpopt(TruckTrailerTime(), 50, "truck_trailer_time")[2])
    print("solving HangingChain2DMPC with Fatrop")
    stats_fatrop.append(SolveFatrop(HangingChain2DMPC(), 25, "hanging_chain_2d_mpc")[2])
    # print("solving HangingChain2DMPC with Ipopt")
    # stats_ipopt.append(SolveIpopt(HangingChain2DMPC(), 40, "hanging_chain_2d_mpc")[2])
    stats_ipopt.append(SolverStats())
    print("solving HangingChain3DMPC with Fatrop")
    stats_fatrop.append(SolveFatrop(HangingChain3DMPC(), 25, "hanging_chain_3d_mpc")[2])
    # print("solving HangingChain3DMPC with Ipopt")
    # stats_ipopt.append(SolveIpopt(HangingChain3DMPC(), 40, "hanging_chain_3d_mpc")[2])
    stats_ipopt.append(SolverStats())
    # print("solving MoonLanderOptTime with Fatrop")
    # stats_fatrop.append(SolveFatrop(MoonLanderOptTime(), 50, "moonlander_opt_time")[2])
    # print("solving MoonLanderOptTime with Ipopt")
    # stats_ipopt.append(SolveIpopt(MoonLanderOptTime(), 50, "moonlander_opt_time")[2])
    ###################### MPC problems with ACADOS ######################
    # cart pendulum 
    qp_solver_cond_list = [1, 2, 3, 4, 5, 6, 8, 12, 25]
    print("solving CartPendulumMPC with Acados")
    stats_acados += [SolveAcados(CartPendulumMPC(), 25, name = "cart_pendulum_mpc", mode = 'SQP', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    stats_acados += [SolveAcados(CartPendulumMPC(), 25, name ="cart_pendulum_mpc", mode = 'SQP_RTI', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    # # quadcopter
    print("solving QuadcopterMPC with Acados")
    stats_acados += [SolveAcados(QuadCopterMPC(), 25, name = "quadcopter_mpc", mode = 'SQP', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    stats_acados += [SolveAcados(QuadCopterMPC(), 25, name = "quadcopter_mpc", mode = 'SQP_RTI', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    # # 2D hanging chain
    print("solving HangingChain2DMPC with Acados")
    stats_acados += [SolveAcados(HangingChain2DMPC(), 25, name = "hanging_chain_2D_mpc", mode = 'SQP', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    stats_acados += [SolveAcados(HangingChain2DMPC(), 25, name = "hanging_chain_2D_mpc", mode = 'SQP_RTI', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    # # 3D hanging chain
    print("solving HangingChain3DMPC with Acados")
    stats_acados += [SolveAcados(HangingChain3DMPC(), 25, name = "hanging_chain_3D_mpc", mode = 'SQP', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    stats_acados += [SolveAcados(HangingChain3DMPC(), 25, name = "hanging_chain_3D_mpc", mode = 'SQP_RTI', qp_solver_cond=pcondi)[2] for pcondi in qp_solver_cond_list]
    for i in range(len(stats_fatrop)):
        print(stats_fatrop[i])
        print(stats_ipopt[i])
    for i in range(len(stats_acados)):
        print(stats_acados[i])
    print_header()
    for i in range(len(stats_fatrop)):
        print_line(stats_fatrop[i], stats_ipopt[i])