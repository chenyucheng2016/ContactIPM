import rospy
import numpy as np
from sensor_msgs.msg import JointState

class JointStatePublisher:
    def __init__(self, topic_name, joint_names):
        rospy.init_node('joint_state_publisher', anonymous=True)
        self.no_states = len(joint_names)
        self.joint_state_pub = rospy.Publisher(topic_name, JointState, queue_size=10)
        self.joint_state = JointState()
        self.joint_state.name = joint_names 
        self.joint_state.position = np.zeros(self.no_states) 
        self.joint_state.velocity = np.zeros(self.no_states) 
        self.joint_state.effort = np.zeros(self.no_states) 

    def publish(self):
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state_pub.publish(self.joint_state)
    
    def set_joint_state(self, joint_state):
        self.joint_state.position = joint_state
    
    def publish_joint_state(self, joint_state):
        self.set_joint_state(joint_state)
        self.publish()
    
    def publish_loop(self, joint_states_array, rate=25):
        rate = rospy.Rate(rate)
        N = joint_states_array.shape[0]
        for i in range(N):
            self.publish_joint_state(joint_states_array[i, :])
            rate.sleep()