## this file contains the code to publish the state of the franka robot to ros topic
from RosJointStatePublisher import JointStatePublisher
from benchmark_tools import SolveFatrop, SolveIpopt
from one_obstacle.problem_specificationT1 import RobotObs
from visualization_msgs.msg import Marker
import rospy
class ObstaclePublisher:
    def __init__(self):
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        self.marker = Marker()

        self.marker.header.frame_id = "world"

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        self.marker.type = 2
        self.marker.id = 0

    def set_spec(self, obstacle_pos, rad):

        # Set the scale of the marker
        self.marker.scale.x = 2*rad 
        self.marker.scale.y = 2*rad 
        self.marker.scale.z = 2*rad 

        # Set the color
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 0.8

        # Set the pose of the marker
        self.marker.pose.position.x = obstacle_pos[0] 
        self.marker.pose.position.y = obstacle_pos[1] 
        self.marker.pose.position.z = obstacle_pos[2] 
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        return self
    def publish(self):
        self.marker.header.stamp = rospy.Time.now()
        self.marker_pub.publish(self.marker)


if __name__ == '__main__':
    N = 50 
    problem = RobotObs()
    # res, prob, _ = SolveFatrop(problem , N)
    res, prob, _ = SolveIpopt(problem , N, jit = False)
    joint_res = res.sample(prob.joints_theta, grid = 'control')[1]
    publish = JointStatePublisher("/joint_states", ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"])
    publish_obstacle = ObstaclePublisher().set_spec(problem.obstacle_pos, problem.obstacle_radius)
    input("Press Enter to publish first joint state...")
    publish_obstacle.publish()
    publish.publish_joint_state(joint_res[0, :])
    input("Press Enter to start publishing...")
    publish.publish_loop(joint_res, rate = 10)
