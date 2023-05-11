#!/usr/bin/env python
import time
import rospy
import roslib
import os
import actionlib
import cv2
import rosbag
import threading
import queue
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from bearnav2.msg import MapRepeaterAction, MapRepeaterResult, SensorsInput, SensorsOutput, ImageList, FeaturesList, \
    Features
from bearnav2.srv import SetDist, SetClockGain, SetClockGainResponse, Alignment, Representations
import numpy as np
import ros_numpy


def parse_camera_msg(msg):
    img = ros_numpy.numpify(msg)
    if "bgr" in msg.encoding:
        img = img[..., ::-1]  # switch from bgr to rgb
    img_msg = ros_numpy.msgify(Image, img, "rgb8")
    return img_msg


def load_map(mappaths, images, distances):
    if "," in mappaths:
        mappaths = mappaths.split(",")
    else:
        mappaths = [mappaths]
    for map_idx, mappath in enumerate(mappaths):
        tmp = []
        for file in list(os.listdir(mappath)):
            if file.endswith(".npy"):
                tmp.append(file[:-4])
        rospy.logwarn(str(len(tmp)) + " images found in the map")
        tmp.sort(key=lambda x: float(x))
        tmp_images = []
        tmp_distances = []

        for idx, dist in enumerate(tmp):
            tmp_distances.append(float(dist))
            with open(os.path.join(mappath, dist + ".npy"), 'rb') as fp:
                map_point = np.load(fp, allow_pickle=True, fix_imports=False).item(0)
                r = map_point["representation"]
                feature = Features()
                feature.shape = r.shape
                feature.values = list(r.flatten())
                tmp_images.append(feature)
                rospy.loginfo("Loaded feature: " + dist + str(".npy"))
        images.append(tmp_images)
        distances.append(tmp_distances)
        rospy.logwarn("Whole map " + str(mappath) + " sucessfully loaded")


class ActionServer():

    def __init__(self):

        # some vars
        self.img = None
        self.mapName = ""
        self.mapStep = None
        self.nextStep = 0
        self.bag = None
        self.isRepeating = False
        self.endPosition = 1.0
        self.clockGain = 1.0
        self.curr_dist = 0.0
        self.map_images = []
        self.map_distances = []
        self.action_dists = None
        self.map_publish_span = 1
        self.use_distances = True
        self.distance_finish_offset = 0.2
        self.map_num = 0
        self.nearest_map_img = -1
        self.curr_map = 0

        rospy.logdebug("Waiting for services to become available...")
        rospy.wait_for_service("repeat/set_dist")
        rospy.wait_for_service("repeat/set_align")
        rospy.Service('set_clock_gain', SetClockGain, self.setClockGain)

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("repeat/set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("repeat/set_align", SetDist)
        self.distance_sub = rospy.Subscriber("repeat/output_dist", SensorsOutput, self.distanceCB, queue_size=1)

        rospy.logdebug("Connecting to sensors module")
        self.sensors_pub = rospy.Publisher("map_representations", SensorsInput, queue_size=1)

        rospy.logdebug("Setting up published for commands")
        self.joy_sub = rospy.Subscriber("/joy_secondary/joy", Joy, self.play_action)

        self.joy_topic = "map_vel"
        self.joy_pub = rospy.Publisher(self.joy_topic, Twist, queue_size=1)

        rospy.logdebug("Starting repeater server")
        self.server = actionlib.SimpleActionServer("repeater", MapRepeaterAction, execute_cb=self.actionCB,
                                                   auto_start=False)
        self.server.register_preempt_callback(self.checkShutdown)
        self.server.start()

        rospy.logwarn("Repeater started, awaiting goal")

    def setClockGain(self, req):
        self.clockGain = req.gain
        return SetClockGainResponse()

    def pubSensorsInput(self):
        # rospy.logwarn("Obtained image!")
        if not self.isRepeating:
            return
        if len(self.map_images) > 0:
            # rospy.logwarn(self.map_distances)
            # Load data from each map the map
            features = []
            distances = []
            last_nearest_img = self.nearest_map_img
            for map_idx in range(self.map_num):
                self.nearest_map_img = np.argmin(abs(self.curr_dist - np.array(self.map_distances[map_idx])))
                # allow only move in map by one image per iteration
                lower_bound = max(0, self.nearest_map_img - self.map_publish_span)
                upper_bound = min(self.nearest_map_img + self.map_publish_span + 1, len(self.map_distances[map_idx]))

                features.extend(self.map_images[map_idx][lower_bound:upper_bound])
                distances.extend(self.map_distances[map_idx][lower_bound:upper_bound])
            if self.nearest_map_img != last_nearest_img:
                rospy.loginfo("matching image " + str(self.map_distances[-1][self.nearest_map_img]) +
                              " at distance " + str(self.curr_dist))
            # Create message for estimators
            sns_in = SensorsInput()
            sns_in.header.stamp = rospy.Time.now()
            sns_in.live_features = []
            sns_in.map_features = features
            sns_in.map_distances = distances

            # rospy.logwarn("message created")
            self.sensors_pub.publish(sns_in)

    def distanceCB(self, msg):
        if self.isRepeating == False:
            return

        # if self.img is None:
        #     rospy.logwarn("Warning: no image received")

        self.curr_dist = msg.output

        if self.curr_dist >= (
                self.map_distances[self.curr_map][-1] - self.distance_finish_offset) and self.use_distances or \
                (self.endPosition != 0.0 and self.endPosition < self.curr_dist):
            rospy.logwarn("GOAL REACHED, STOPPING REPEATER")
            self.isRepeating = False
            if self.use_distances:
                self.action_dists = []
                self.actions = []
            self.shutdown()

        self.pubSensorsInput()

    def goalValid(self, goal):

        if goal.mapName == "":
            rospy.logwarn("Goal missing map name")
            return False
        # if not os.path.isdir(goal.mapName):
        #     rospy.logwarn("Can't find map directory")
        #     return False
        # if not os.path.isfile(os.path.join(goal.mapName, goal.mapName + ".bag")):
        #     rospy.logwarn("Can't find commands")
        #     return False
        # if not os.path.isfile(os.path.join(goal.mapName, "params")):
        #     rospy.logwarn("Can't find params")
        #     return False
        if goal.startPos < 0:
            rospy.logwarn("Invalid (negative) start position). Use zero to start at the beginning")
            return False
        if goal.startPos > goal.endPos:
            rospy.logwarn("Start position greater than end position")
            return False
        return True

    def actionCB(self, goal):

        rospy.loginfo("New goal received")
        result = MapRepeaterResult()
        if self.goalValid(goal) == False:
            rospy.logwarn("Ignoring invalid goal")
            result.success = False
            self.server.set_succeeded(result)
            return

        map_name = goal.mapName.split(",")[0]
        self.parseParams(os.path.join(map_name, "params"))

        self.map_publish_span = int(goal.imagePub)

        # set distance to zero
        rospy.logdebug("Resetting distnace and alignment")
        self.align_reset_srv(0.0, 1)
        self.endPosition = goal.endPos
        self.nextStep = 0

        # reload all the buffers
        self.map_images = []
        self.map_distances = []
        self.action_dists = None

        map_loader = threading.Thread(target=load_map, args=(goal.mapName, self.map_images, self.map_distances))
        map_loader.start()
        # TODO: Here we are waiting for th thread to join, so it does not make sense to use separate thread
        map_loader.join()
        self.map_num = 1

        rospy.logwarn("Starting repeat")
        self.bag = rosbag.Bag(os.path.join(map_name, map_name + ".bag"), "r")
        self.mapName = goal.mapName

        self.distance_reset_srv(goal.startPos, self.map_num)
        self.curr_dist = goal.startPos
        time.sleep(2)  # waiting till some map images are parsed
        self.isRepeating = True
        # kick into the robot at the beggining:
        rospy.loginfo("Repeating started!")

        # self.shutdown() only for sth
        result.success = True
        self.server.set_succeeded(result)

    def parseParams(self, filename):

        with open(filename, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = filter(None, data)
        for line in data:
            line = line.split(" ")
            if "stepSize" in line[0]:
                rospy.logdebug("Setting step size to: %s" % (line[1]))
                self.mapStep = float(line[1])
            if "odomTopic" in line[0]:
                rospy.logdebug("Saved odometry topic is: %s" % (line[1]))
                self.savedOdomTopic = line[1]

    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isRepeating = False
        if self.bag is not None:
            self.bag.close()

    def play_action(self, msg):
        forward_action = msg.axes[1]
        out = Twist()
        out.linear.x = forward_action
        if self.isRepeating:
            self.joy_pub.publish(out)


if __name__ == '__main__':
    rospy.init_node("replayer_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()
