#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from dofbot_pro_KCF.Dofbot_Track import *
from dofbot_pro_interface.msg import *
from std_msgs.msg import Float32,Bool,Int8

class KCFTrackingNode(Node):
    def __init__(self):
        super().__init__('KCF_tracking')

        self.dofbot_tracker = DofbotTrack()

        self.pos_sub = Subscriber(self,Position,'/pos_xyz')
        self.time_sync = ApproximateTimeSynchronizer([self.pos_sub],queue_size=10,slop =0.5,allow_headerless=True)
        self.time_sync.registerCallback(self.TrackAndGrap)

        self.sub_voice = self.create_subscription(Int8,"voice_result",self.getVoiceResultCallBack,1)
        self.cur_distance = 0.0
        self.cnt = 0
        self.start_flag = False
        print("Init done!")
                        
    
    def TrackAndGrap(self,position):  

        if self.start_flag == True :
            center_x, center_y = position.x,position.y
            self.cur_distance = position.z
            # print("---------------------")
            if abs(center_x-320) >8 or abs(center_y-240)>8 :
                self.dofbot_tracker.XY_track(center_x,center_y)         
            else:
                self.cnt = self.cnt + 1
                if self.cnt == 10 :
                    self.cnt = 0
                    if self.cur_distance!= 999:
                        self.dofbot_tracker.stop_flag = True
                        self.dofbot_tracker.Clamping(center_x,center_y,self.cur_distance)

    def getVoiceResultCallBack(self,msg):
        if msg.data == 14:
            self.start_flag = True
            print("Start tracking and grabbing.")

def main(args=None):
    rclpy.init(args=args)
    kcf_tracking = KCFTrackingNode()
    kcf_tracking.dofbot_tracker.pub_arm(kcf_tracking.dofbot_tracker.init_joints)
    try:    
        rclpy.spin(kcf_tracking)
    except KeyboardInterrupt:
        pass
    finally:
        kcf_tracking.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 

