## this file contains the code to publish the state of the franka robot to ros topic
from RosJointStatePublisher import JointStatePublisher
from benchmark_tools import SolveFatrop, SolveIpopt
from p2p.problem_specificationT1 import QuadCopterP2P  
from three_obstacles.problem_specificationT1 import QuadCopterThreeObs  
from one_obstacle.problem_specificationT1 import QuadCopterOneObs
from mpc.problem_specification import QuadCopterMPC
from visualization_msgs.msg import Marker
import casadi as cs
import rospy
import time
class ObstaclePublisher:
    def __init__(self):
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)


        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        # self.marker.id = 0

    def set_spec(self, id, obstacle_pos, rad, orientation):
        self.marker = Marker()
        self.marker.header.frame_id = "world"
        self.marker.type = 3
        self.marker.id = id

        # Set the scale of the marker
        self.marker.scale.x = 2*rad  -0.8
        self.marker.scale.y = 2*rad  -0.8
        self.marker.scale.z = 15 

        # Set the color
        if id == 0:
            self.marker.color.r = 0.0
            self.marker.color.g = 1.0
            self.marker.color.b = 0.0
            self.marker.color.a = 0.5
        elif id == 1:
            self.marker.color.r = 1.0
            self.marker.color.g = 0.0
            self.marker.color.b = 0.0
            self.marker.color.a = 0.5
        elif id == 2:
            self.marker.color.r = 0.0
            self.marker.color.g = 0.0
            self.marker.color.b = 1.0
            self.marker.color.a = 0.5
        if orientation == "x":
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.0
            self.marker.pose.orientation.z = 0.0
            self.marker.pose.orientation.w = 1.0
        elif orientation == "z":
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.0
            self.marker.pose.orientation.z = 0.707
            self.marker.pose.orientation.w = 0.707
        elif orientation == "y":
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.707
            self.marker.pose.orientation.z = 0.0
            self.marker.pose.orientation.w = 0.707

        # Set the pose of the marker
        self.marker.pose.position.x = obstacle_pos[0] 
        self.marker.pose.position.y = obstacle_pos[1] 
        self.marker.pose.position.z = obstacle_pos[2] 
        return self
    def publish(self):
        time.sleep(0.01)
        self.marker.header.stamp = rospy.Time.now()
        self.marker_pub.publish(self.marker)
if __name__ == '__main__':
    mode = "mpc"
    if mode == "p2p":
        N = 25
        problem = QuadCopterP2P()
    elif mode == "three_obs":
        N = 100 
        problem = QuadCopterThreeObs()
    elif mode == "one_obs":
        N = 50
        problem = QuadCopterOneObs()
    elif mode == "mpc":
        N = 25
        problem = QuadCopterMPC()
    res, prob, _ = SolveIpopt(problem , N, jit = False)
    joints_sym = cs.vertcat(prob.p, prob.phi , prob.theta ,prob.psi)
    joint_res = res.sample(joints_sym, grid = 'control')[1]
    publish = JointStatePublisher("/joint_states", ["joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"])
    obstacle_vis = ObstaclePublisher()
    input("Press Enter to publish first joint state...")
    if mode == "three_obs":
        obstacle_vis.set_spec(0,problem.p_obs1,  problem.r_obs1, problem.or_pobs1).publish()
        obstacle_vis.set_spec(1,problem.p_obs2, problem.r_obs2, problem.or_pobs2).publish()
        obstacle_vis.set_spec(2,problem.p_obs3,  problem.r_obs3, problem.or_pobs3).publish()
    if mode == "one_obs":
        obstacle_vis.set_spec(0,problem.p_obs1,  problem.r_obs1, problem.or_obs1).publish()
    publish.publish_joint_state(joint_res[0, :])
    input("Press Enter to start publishing...")
    publish.publish_loop(joint_res, rate = N/2.0)