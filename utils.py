import numpy as np
import cv2
import ctypes
import os, inspect, sys
from multiprocessing import Process as Thread
import json
import multiprocessing.queues as queue
import shutil
import matplotlib.pyplot as plt

# src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# # Windows and Linux
# arch_dir = './leap_lib'
# # Mac
# #arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
#
# sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
#
# from . import Leap


def draw_demo_ui(detected_gesture, img0=None, img1=None, text='RUNNING...'):

    position = (10, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0  # 1
    fontColor = (0xFFFF, 0xFFFF, 0xFFFF)
    fontColorTrue = (1028, 60909, 6939)# (0x05A0, 0x05A0, 0x05A0)
    lineType = 1

    img = np.zeros((700, 1000, 3), np.uint16)

    cv2.putText(img, "DEPTH MAP",
                (108, 356),
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(img, "IR IMAGE",
                (108, 40,),
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(img, text,
                (500, 300),
                font,
                fontScale*1.5,
                fontColor if text == 'RUNNING...' else fontColorTrue,
                lineType)

    if img0 is not None:
        img[50:306, 10:346, :] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

        img[376:632, 10:346, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # cv2.circle(img, (700, 100), 50, color=(255, 0, 0), thickness=-1)

    cv2.imshow('DEMO', img)
    cv2.moveWindow('DEMO', 100, 0)


def draw_ui(text, circle=False, thickness=1, position=(0, 50)):

    position = (0, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5 #1
    fontColor = (0, 255, 255)
    lineType = 1

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.putText(img, text,
                position,
                font,
                fontScale,
                fontColor,
                lineType)
    if circle:
        cv2.circle(img, (700, 100), 50, color=(255, 0, 0), thickness=thickness)

    cv2.imshow('', img)
    # cv2.waitKey(1)


#############################
#        SAVING THREADS     #
#############################

def save_single_record(list_rr, list_ru, list_lr, list_lu, list_json_obj, list_img_rgb,
                 list_img_z, list_img_ir, dir_rr, dir_ru, dir_lr, dir_lu, dir_leap_info, dir_rgb, dir_z, dir_ir):

    if not os.path.exists(dir_rr) and not os.path.exists(dir_lr) and \
            not os.path.exists(dir_lr) and not os.path.exists(dir_lu) \
            and not os.path.exists(dir_leap_info) and not os.path.exists(dir_rgb) \
            and not os.path.exists(dir_z) and not os.path.exists(dir_ir):
        os.makedirs(dir_rr)
        os.makedirs(dir_lr)
        os.makedirs(dir_ru)
        os.makedirs(dir_lu)
        os.makedirs(dir_leap_info)
        os.makedirs(dir_rgb)
        os.makedirs(dir_z)
        os.makedirs(dir_ir)

        for i, (img_rr, img_ru, img_lr, img_lu, json_obj, img_rgb, img_z, img_ir) in enumerate(zip(list_rr,
                                                                                                   list_ru,
                                                                                                   list_lr,
                                                                                                   list_lu,
                                                                                                   list_json_obj,
                                                                                                   list_img_rgb,
                                                                                                   list_img_z,
                                                                                                   list_img_ir)):

            if img_ru is not None:
                cv2.imwrite("{}/{:03d}_ru.png".format(dir_ru, i), img_ru)
            if img_rr is not None:
                cv2.imwrite("{}/{:03d}_rr.png".format(dir_rr, i), img_rr)
            if img_lu is not None:
                cv2.imwrite("{}/{:03d}_ul.png".format(dir_lu, i), img_lu)
            if img_lr is not None:
                cv2.imwrite("{}/{:03d}_rl.png".format(dir_lr, i), img_lr)
            if img_rgb is not None:
                cv2.imwrite("{}/{:03d}_rgb.png".format(dir_rgb, i), img_rgb)
            if img_z is not None:
                np.savetxt("{}/{:03d}_z.gz".format(dir_z, i), img_z)
            if img_ir is not None:
                cv2.imwrite("{}/{:03d}_ir.png".format(dir_ir, i), img_ir)

            # print('ok')
            if json_obj is not None:
                with open("{}/{:03d}_js.json".format(dir_leap_info, i), 'w') as outfile:
                    json.dump(json_obj, outfile)

        print("session saved")


class GestureData:

    def __init__(self, id_gesture, list_rr, list_ru, list_lr, list_lu, list_json_obj, list_img_rgb,
                 list_img_z, list_img_ir, dir_rr, dir_ru, dir_lr, dir_lu, dir_leap_info, dir_rgb, dir_z, dir_ir,
                 rewrite=False):

        self.id_gesture = id_gesture
        self.rewrite = rewrite
        self.list_img_rr = list_rr
        self.list_img_ru = list_ru
        self.list_img_lr = list_lr
        self.list_img_lu = list_lu
        self.list_img_rgb = list_img_rgb
        self.list_img_z = list_img_z
        self.list_img_ir = list_img_ir
        self.list_json = list_json_obj
        self.directory_ru = dir_ru
        self.directory_rr = dir_rr
        self.directory_lu = dir_lu
        self.directory_lr = dir_lr
        self.directory_leap_info = dir_leap_info
        self.directory_rgb = dir_rgb
        self.directory_z = dir_z
        self.directory_ir = dir_ir

        if not rewrite:
            if not os.path.exists(self.directory_rr)and not os.path.exists(self.directory_lr) and \
                    not os.path.exists(self.directory_lr) and not os.path.exists(self.directory_lu) \
                    and not os.path.exists(self.directory_leap_info) and not os.path.exists(self.directory_rgb) \
                    and not os.path.exists(self.directory_z) and not os.path.exists(self.directory_ir):
                os.makedirs(self.directory_rr)
                os.makedirs(self.directory_lr)
                os.makedirs(self.directory_ru)
                os.makedirs(self.directory_lu)
                os.makedirs(self.directory_leap_info)
                os.makedirs(self.directory_rgb)
                os.makedirs(self.directory_z)
                os.makedirs(self.directory_ir)

            else:
                print("error on loading session info")
                exit(-1)
        else:
            # remove old directories and files
            shutil.rmtree(self.directory_rr, ignore_errors=True)
            shutil.rmtree(self.directory_lr, ignore_errors=True)
            shutil.rmtree(self.directory_ru, ignore_errors=True)
            shutil.rmtree(self.directory_lu, ignore_errors=True)
            shutil.rmtree(self.directory_leap_info, ignore_errors=True)
            shutil.rmtree(self.directory_rgb, ignore_errors=True)
            shutil.rmtree(self.directory_z, ignore_errors=True)
            shutil.rmtree(self.directory_ir, ignore_errors=True)

            # create new directories

            if not os.path.exists(self.directory_rr) and not os.path.exists(self.directory_lr) and \
                    not os.path.exists(self.directory_lr) and not os.path.exists(self.directory_lu) \
                    and not os.path.exists(self.directory_leap_info) and not os.path.exists(self.directory_rgb) \
                    and not os.path.exists(self.directory_z) and not os.path.exists(self.directory_ir):
                os.makedirs(self.directory_rr)
                os.makedirs(self.directory_lr)
                os.makedirs(self.directory_ru)
                os.makedirs(self.directory_lu)
                os.makedirs(self.directory_leap_info)
                os.makedirs(self.directory_rgb)
                os.makedirs(self.directory_z)
                os.makedirs(self.directory_ir)

            else:
                print("error on loading session info")
                exit(-1)

    def saveGestureData(self):

        th = ThreadWritingGesture(self.id_gesture, self.list_img_rr, self.list_img_ru, self.list_img_lr,
                                  self.list_img_lu, self.list_json,
                                  self.list_img_rgb, self.list_img_z, self.list_img_ir, self.directory_rr,
                                  self.directory_ru, self.directory_lr, self.directory_lu, self.directory_leap_info,
                                  self.directory_rgb, self.directory_z, self.directory_ir)

        th.start()
        return th


class ThreadWritingGesture(Thread):

    def __init__(self, id_gesture, list_rr, list_ru, list_lr, list_lu, list_json_obj, list_img_rgb,
                 list_img_z, list_img_ir, dir_rr, dir_ru, dir_lr, dir_lu, dir_leap_info, dir_rgb, dir_z, dir_ir):

        Thread.__init__(self)
        self.id_gesture = id_gesture
        self.list_img_rr = list_rr
        self.list_img_ru = list_ru
        self.list_img_lr = list_lr
        self.list_img_lu = list_lu
        self.list_img_rgb = list_img_rgb
        self.list_img_z = list_img_z
        self.list_img_ir = list_img_ir
        self.list_json = list_json_obj
        self.directory_ru = dir_ru
        self.directory_rr = dir_rr
        self.directory_lu = dir_lu
        self.directory_lr = dir_lr
        self.directory_leap_info = dir_leap_info
        self.directory_rgb = dir_rgb
        self.directory_z = dir_z
        self.directory_ir = dir_ir

    def run(self):
        print("saving gesture {}".format(self.id_gesture))
        # print(len(self.list_img_rr), len(self.list_img_ru), len(self.list_img_lr), len(self.list_img_lu),
        #       len(self.list_json))

        # cut list img

        for i, (img_rr, img_ru, img_lr, img_lu, json_obj, img_rgb, img_z, img_ir) in enumerate(zip(self.list_img_rr,
                                                                                    self.list_img_ru,
                                                                                    self.list_img_lr,
                                                                                    self.list_img_lu,
                                                                                    self.list_json,
                                                                                    self.list_img_rgb,
                                                                                    self.list_img_z,
                                                                                    self.list_img_ir)):

            if img_ru is not None:
                cv2.imwrite("{}/{:03d}_ru.png".format(self.directory_ru, i), img_ru)
            if img_rr is not None:
                cv2.imwrite("{}/{:03d}_rr.png".format(self.directory_rr, i), img_rr)
            if img_lu is not None:
                cv2.imwrite("{}/{:03d}_ul.png".format(self.directory_lu, i), img_lu)
            if img_lr is not None:
                cv2.imwrite("{}/{:03d}_rl.png".format(self.directory_lr, i), img_lr)
            if img_rgb is not None:
                cv2.imwrite("{}/{:03d}_rgb.png".format(self.directory_rgb, i), img_rgb)
            if img_z is not None:
                np.savetxt("{}/{:03d}_z.gz".format(self.directory_z, i), img_z)
            if img_ir is not None:
                cv2.imwrite("{}/{:03d}_ir.png".format(self.directory_ir, i), img_ir)

            # print('ok')
            if json_obj is not None:
                with open("{}/{:03d}_js.json".format(self.directory_leap_info, i), 'w') as outfile:
                    json.dump(json_obj, outfile)

        print('saving gesture {} completed'.format(self.id_gesture))


class ThreadOnDisk(Thread):

    def __init__(self, img_rr, img_ru, img_lr, img_lu, json_obj, img_rgb, img_z, img_ir,
                 frame_counter, directory_rr,
                 directory_ru, directory_lr, directory_lu, directory_leap_info, directory_rgb, directory_z,
                 directory_ir):

        Thread.__init__(self)
        self.img_rr = img_rr
        self.img_ru = img_ru
        self.img_lr = img_lr
        self.img_lu = img_lu
        self.json_obj = json_obj
        self.img_rgb = img_rgb
        self.img_z = img_z
        self.img_ir = img_ir
        self.frame_counter = frame_counter
        self.directory_rr = directory_rr
        self.directory_ru = directory_ru
        self.directory_lr = directory_lr
        self.directory_lu = directory_lu
        self.directory_leap_info = directory_leap_info
        self.directory_rgb = directory_rgb
        self.directory_z = directory_z
        self.directory_ir = directory_ir

    def run(self):
        cv2.imwrite("{0}/{1}_ru.png".format(self.directory_ru, self.frame_counter), self.img_ru)
        cv2.imwrite("{0}/{1}_lu.png".format(self.directory_lu, self.frame_counter), self.img_lu)
        # write raw
        cv2.imwrite("{0}/{1}_rr.png".format(self.directory_rr, self.frame_counter), self.img_rr)
        cv2.imwrite("{0}/{1}_lr.png".format(self.directory_lr, self.frame_counter), self.img_lr)
        # write rgb
        cv2.imwrite("{0}/{1}_rgb.png".format(self.directory_rgb, self.frame_counter), self.img_rgb)
        # depth
        np.savetxt("{0}/{1}_z.gz".format(self.directory_z, self.frame_counter), self.img_z)
        cv2.imwrite("{0}/{1}_ir.png".format(self.directory_ir, self.frame_counter), self.img_ir)

        #
        with open("{0}/{1}_js.json".format(self.directory_leap_info, self.frame_counter), 'w') as outfile:
            json.dump(self.json_obj, outfile)

#############################
#        SESSION INFO       #
#############################


file_info = "session_info.json"


def save_session_info(session_id):

    session_info = {

        'id': session_id
    }

    with open(file_info, 'w') as outfile:
        json.dump(session_info, outfile)


def load_session_info():

    with open(file_info) as infile:
        info = json.load(infile)

    return info["id"]

#############################
#         PICOFLEXX         #
#############################


def process_event_queue(q):
    # create a loop that will run for the given amount of time

    try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
           # or the timeout of 1 second is hit
        # print('ok')
        # bloccante con timeout
        item = q.get(True, 1)
        # item = q.get(False)
    except queue.Empty:
        # this will be thrown when the timeout is hit
        print("error in queue")
        return None
    else:
        return item


def get_images_from_picoflexx(queue):

    item = process_event_queue(queue)
    if item is not None:
        # print(" item secondno noi NON None", item)
        z = item[0]
        g = item[1]

        g = g.astype(np.uint16)
        g = cv2.normalize(g, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

        ret = True
        return ret, (z, g)
    else:
        # print(" item secondno noi None", item)
        ret = False
        return ret, (None, None)

#############################
#    LEAP MOTION IMAGES     #
#############################


def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length//2, dtype=np.float32)
    ymap = np.zeros(distortion_length//2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length//2 - i//2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length//2 - i//2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width//2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width//2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination


def get_raw_image(image):
    image_buffer_ptr = image.data_pointer
    ctype_array_def = ctypes.c_ubyte * image.width * image.height
    as_ctype_array = ctype_array_def.from_address(int(image_buffer_ptr))
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    return as_numpy_array


# function needed to validate the start of recording (OPTIONAL)
def hand_is_valid(frame):

    hand = frame.hands[0]
    return hand.is_right and hand.is_valid

#############################
#         LEAP MOTION       #
#############################

#right hand from frame.hands
def frame2json_struct(frame):

    j_frame = {}
    f = None
    if frame.is_valid:
        f = frame
    else:
        j_frame['frame'] = 'invalid'

    h = None
    hand = frame.hands[0]
    if hand.is_right and hand.is_valid:
        h = hand

    if h is None:
        j_frame['frame'] = 'invalid'
        return j_frame

    fingers_list = []
    for i in range(len(h.fingers)):
        # fin = h.fingers
        # print(fin)
        fingers_list.append(h.fingers[i])

    pointables_list = []
    for i in range(len(h.pointables)):
        pointables_list.append(h.pointables[i])

    bones = {
        't': {
            'metacarpal': fingers_list[0].bone(0),
            'proximal':  fingers_list[0].bone(1),
            'intermediate': fingers_list[0].bone(2),
            'distal': fingers_list[0].bone(3)
        },
        'i': {
            'metacarpal': fingers_list[1].bone(0),
            'proximal': fingers_list[1].bone(1),
            'intermediate': fingers_list[1].bone(2),
            'distal': fingers_list[1].bone(3)
        },
        'm': {
            'metacarpal': fingers_list[2].bone(0),
            'proximal': fingers_list[2].bone(1),
            'intermediate': fingers_list[2].bone(2),
            'distal': fingers_list[2].bone(3)
        },
        'r': {
            'metacarpal': fingers_list[3].bone(0),
            'proximal': fingers_list[3].bone(1),
            'intermediate': fingers_list[3].bone(2),
            'distal': fingers_list[3].bone(3)
        },
        'p': {
            'metacarpal': fingers_list[4].bone(0),
            'proximal': fingers_list[4].bone(1),
            'intermediate': fingers_list[4].bone(2),
            'distal': fingers_list[4].bone(3)
        }
    }

    # costruzione json

    j_frame['frame'] = {
        'id': f.id,
        'timestamp': f.timestamp,
        'right_hand': {
            'id': h.id,
            'palm_position': [h.palm_position.x, h.palm_position.y, h.palm_position.z, h.palm_position.pitch, h.palm_position.yaw, h.palm_position.roll],
            'palm_normal': [h.palm_normal.x, h.palm_normal.y, h.palm_normal.z, h.palm_normal.pitch, h.palm_normal.yaw, h.palm_normal.roll],
            'palm_velocity': [h.palm_velocity.x, h.palm_velocity.y, h.palm_velocity.z, h.palm_velocity.pitch, h.palm_velocity.yaw, h.palm_velocity.roll],
            'palm_width': h.palm_width,
            'pinch_strength': h.pinch_strength,
            'grab_strength': h.grab_strength,
            'direction': [h.direction.x, h.direction.y, h.direction.z, h.direction.pitch, h.direction.yaw, h.direction.roll],
            'sphere_center': [h.sphere_center.x, h.sphere_center.y, h.sphere_center.z, h.sphere_center.pitch, h.sphere_center.yaw, h.sphere_center.roll],
            'sphere_radius': h.sphere_radius,
            'wrist_position': [h.wrist_position.x, h.wrist_position.y, h.wrist_position.z, h.wrist_position.pitch, h.wrist_position.yaw, h.wrist_position.roll],
            'fingers': {
                'thumb': {
                    'id': fingers_list[0].id,
                    'length': fingers_list[0].length,
                    'width': fingers_list[0].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['t']['metacarpal'].center.x, bones['t']['metacarpal'].center.y, bones['t']['metacarpal'].center.z,
                                       bones['t']['metacarpal'].center.pitch, bones['t']['metacarpal'].center.yaw, bones['t']['metacarpal'].center.roll],
                            'direction': [bones['t']['metacarpal'].direction.x, bones['t']['metacarpal'].direction.y, bones['t']['metacarpal'].direction.z,
                                       bones['t']['metacarpal'].direction.pitch, bones['t']['metacarpal'].direction.yaw, bones['t']['metacarpal'].direction.roll],
                            'length':  bones['t']['metacarpal'].length,
                            'width':  bones['t']['metacarpal'].width,
                            'prev_joint': [bones['t']['metacarpal'].prev_joint.x,bones['t']['metacarpal'].prev_joint.y, bones['t']['metacarpal'].prev_joint.z,
                                           bones['t']['metacarpal'].prev_joint.pitch, bones['t']['metacarpal'].prev_joint.yaw, bones['t']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['t']['metacarpal'].next_joint.x, bones['t']['metacarpal'].next_joint.y, bones['t']['metacarpal'].next_joint.z,
                                           bones['t']['metacarpal'].next_joint.pitch, bones['t']['metacarpal'].next_joint.yaw, bones['t']['metacarpal'].next_joint.roll]

                        },

                        'proximal': {
                            'center': [bones['t']['proximal'].center.x, bones['t']['proximal'].center.y,
                                       bones['t']['proximal'].center.z,
                                       bones['t']['proximal'].center.pitch, bones['t']['proximal'].center.yaw,
                                       bones['t']['proximal'].center.roll],
                            'direction': [bones['t']['proximal'].direction.x, bones['t']['proximal'].direction.y,
                                          bones['t']['proximal'].direction.z,
                                          bones['t']['proximal'].direction.pitch,
                                          bones['t']['proximal'].direction.yaw,
                                          bones['t']['proximal'].direction.roll],
                            'length': bones['t']['proximal'].length,
                            'width': bones['t']['proximal'].width,
                            'prev_joint': [bones['t']['proximal'].prev_joint.x, bones['t']['proximal'].prev_joint.y,
                                           bones['t']['proximal'].prev_joint.z,
                                           bones['t']['proximal'].prev_joint.pitch,
                                           bones['t']['proximal'].prev_joint.yaw,
                                           bones['t']['proximal'].prev_joint.roll],
                            'next_joint': [bones['t']['proximal'].next_joint.x, bones['t']['proximal'].next_joint.y,
                                           bones['t']['proximal'].next_joint.z,
                                           bones['t']['proximal'].next_joint.pitch,
                                           bones['t']['proximal'].next_joint.yaw,
                                           bones['t']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['t']['intermediate'].center.x, bones['t']['intermediate'].center.y,
                                       bones['t']['intermediate'].center.z,
                                       bones['t']['intermediate'].center.pitch, bones['t']['intermediate'].center.yaw,
                                       bones['t']['intermediate'].center.roll],
                            'direction': [bones['t']['intermediate'].direction.x, bones['t']['intermediate'].direction.y,
                                          bones['t']['intermediate'].direction.z,
                                          bones['t']['intermediate'].direction.pitch,
                                          bones['t']['intermediate'].direction.yaw,
                                          bones['t']['intermediate'].direction.roll],
                            'length': bones['t']['intermediate'].length,
                            'width': bones['t']['intermediate'].width,
                            'prev_joint': [bones['t']['intermediate'].prev_joint.x, bones['t']['intermediate'].prev_joint.y,
                                           bones['t']['intermediate'].prev_joint.z,
                                           bones['t']['intermediate'].prev_joint.pitch,
                                           bones['t']['intermediate'].prev_joint.yaw,
                                           bones['t']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['t']['intermediate'].next_joint.x, bones['t']['intermediate'].next_joint.y,
                                           bones['t']['intermediate'].next_joint.z,
                                           bones['t']['intermediate'].next_joint.pitch,
                                           bones['t']['intermediate'].next_joint.yaw,
                                           bones['t']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['t']['distal'].center.x, bones['t']['distal'].center.y,
                                       bones['t']['distal'].center.z,
                                       bones['t']['distal'].center.pitch, bones['t']['distal'].center.yaw,
                                       bones['t']['distal'].center.roll],
                            'direction': [bones['t']['distal'].direction.x, bones['t']['distal'].direction.y,
                                          bones['t']['distal'].direction.z,
                                          bones['t']['distal'].direction.pitch,
                                          bones['t']['distal'].direction.yaw,
                                          bones['t']['distal'].direction.roll],
                            'length': bones['t']['distal'].length,
                            'width': bones['t']['distal'].width,
                            'prev_joint': [bones['t']['distal'].prev_joint.x, bones['t']['distal'].prev_joint.y,
                                           bones['t']['distal'].prev_joint.z,
                                           bones['t']['distal'].prev_joint.pitch,
                                           bones['t']['distal'].prev_joint.yaw,
                                           bones['t']['distal'].prev_joint.roll],
                            'next_joint': [bones['t']['distal'].next_joint.x, bones['t']['distal'].next_joint.y,
                                           bones['t']['distal'].next_joint.z,
                                           bones['t']['distal'].next_joint.pitch,
                                           bones['t']['distal'].next_joint.yaw,
                                           bones['t']['distal'].next_joint.roll]
                        }
                    }

                },

                'index': {
                    'id': fingers_list[1].id,
                    'length': fingers_list[1].length,
                    'width': fingers_list[1].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['i']['metacarpal'].center.x, bones['i']['metacarpal'].center.y,
                                       bones['i']['metacarpal'].center.z,
                                       bones['i']['metacarpal'].center.pitch, bones['i']['metacarpal'].center.yaw,
                                       bones['i']['metacarpal'].center.roll],
                            'direction': [bones['i']['metacarpal'].direction.x, bones['i']['metacarpal'].direction.y,
                                          bones['i']['metacarpal'].direction.z,
                                          bones['i']['metacarpal'].direction.pitch,
                                          bones['i']['metacarpal'].direction.yaw,
                                          bones['i']['metacarpal'].direction.roll],
                            'length': bones['i']['metacarpal'].length,
                            'width': bones['i']['metacarpal'].width,
                            'prev_joint': [bones['i']['metacarpal'].prev_joint.x, bones['i']['metacarpal'].prev_joint.y,
                                           bones['i']['metacarpal'].prev_joint.z,
                                           bones['i']['metacarpal'].prev_joint.pitch,
                                           bones['i']['metacarpal'].prev_joint.yaw,
                                           bones['i']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['i']['metacarpal'].next_joint.x, bones['i']['metacarpal'].next_joint.y,
                                           bones['i']['metacarpal'].next_joint.z,
                                           bones['i']['metacarpal'].next_joint.pitch,
                                           bones['i']['metacarpal'].next_joint.yaw,
                                           bones['m']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['i']['proximal'].center.x, bones['i']['proximal'].center.y,
                                       bones['i']['proximal'].center.z,
                                       bones['i']['proximal'].center.pitch, bones['i']['proximal'].center.yaw,
                                       bones['i']['proximal'].center.roll],
                            'direction': [bones['i']['proximal'].direction.x, bones['i']['proximal'].direction.y,
                                          bones['i']['proximal'].direction.z,
                                          bones['i']['proximal'].direction.pitch,
                                          bones['i']['proximal'].direction.yaw,
                                          bones['i']['proximal'].direction.roll],
                            'length': bones['i']['proximal'].length,
                            'width': bones['i']['proximal'].width,
                            'prev_joint': [bones['i']['proximal'].prev_joint.x, bones['i']['proximal'].prev_joint.y,
                                           bones['i']['proximal'].prev_joint.z,
                                           bones['i']['proximal'].prev_joint.pitch,
                                           bones['i']['proximal'].prev_joint.yaw,
                                           bones['i']['proximal'].prev_joint.roll],
                            'next_joint': [bones['i']['proximal'].next_joint.x, bones['i']['proximal'].next_joint.y,
                                           bones['i']['proximal'].next_joint.z,
                                           bones['i']['proximal'].next_joint.pitch,
                                           bones['i']['proximal'].next_joint.yaw,
                                           bones['i']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['i']['intermediate'].center.x, bones['i']['intermediate'].center.y,
                                       bones['i']['intermediate'].center.z,
                                       bones['i']['intermediate'].center.pitch, bones['i']['intermediate'].center.yaw,
                                       bones['i']['intermediate'].center.roll],
                            'direction': [bones['i']['intermediate'].direction.x, bones['i']['intermediate'].direction.y,
                                          bones['i']['intermediate'].direction.z,
                                          bones['i']['intermediate'].direction.pitch,
                                          bones['i']['intermediate'].direction.yaw,
                                          bones['i']['intermediate'].direction.roll],
                            'length': bones['i']['intermediate'].length,
                            'width': bones['i']['intermediate'].width,
                            'prev_joint': [bones['i']['intermediate'].prev_joint.x, bones['i']['intermediate'].prev_joint.y,
                                           bones['i']['intermediate'].prev_joint.z,
                                           bones['i']['intermediate'].prev_joint.pitch,
                                           bones['i']['intermediate'].prev_joint.yaw,
                                           bones['i']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['i']['intermediate'].next_joint.x, bones['i']['intermediate'].next_joint.y,
                                           bones['i']['intermediate'].next_joint.z,
                                           bones['i']['intermediate'].next_joint.pitch,
                                           bones['i']['intermediate'].next_joint.yaw,
                                           bones['i']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['i']['distal'].center.x, bones['i']['distal'].center.y,
                                       bones['i']['distal'].center.z,
                                       bones['i']['distal'].center.pitch, bones['i']['distal'].center.yaw,
                                       bones['i']['distal'].center.roll],
                            'direction': [bones['i']['distal'].direction.x, bones['i']['distal'].direction.y,
                                          bones['i']['distal'].direction.z,
                                          bones['i']['distal'].direction.pitch,
                                          bones['i']['distal'].direction.yaw,
                                          bones['i']['distal'].direction.roll],
                            'length': bones['i']['distal'].length,
                            'width': bones['i']['distal'].width,
                            'prev_joint': [bones['i']['distal'].prev_joint.x, bones['i']['distal'].prev_joint.y,
                                           bones['i']['distal'].prev_joint.z,
                                           bones['i']['distal'].prev_joint.pitch,
                                           bones['i']['distal'].prev_joint.yaw,
                                           bones['i']['distal'].prev_joint.roll],
                            'next_joint': [bones['i']['distal'].next_joint.x, bones['i']['distal'].next_joint.y,
                                           bones['i']['distal'].next_joint.z,
                                           bones['i']['distal'].next_joint.pitch,
                                           bones['i']['distal'].next_joint.yaw,
                                           bones['i']['distal'].next_joint.roll]
                        }
                    }
                },
                'middle': {
                    'id': fingers_list[2].id,
                    'length': fingers_list[2].length,
                    'width': fingers_list[2].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['m']['metacarpal'].center.x, bones['m']['metacarpal'].center.y,
                                       bones['m']['metacarpal'].center.z,
                                       bones['m']['metacarpal'].center.pitch, bones['m']['metacarpal'].center.yaw,
                                       bones['m']['metacarpal'].center.roll],
                            'direction': [bones['m']['metacarpal'].direction.x, bones['m']['metacarpal'].direction.y,
                                          bones['m']['metacarpal'].direction.z,
                                          bones['m']['metacarpal'].direction.pitch,
                                          bones['m']['metacarpal'].direction.yaw,
                                          bones['m']['metacarpal'].direction.roll],
                            'length': bones['m']['metacarpal'].length,
                            'width': bones['m']['metacarpal'].width,
                            'prev_joint': [bones['m']['metacarpal'].prev_joint.x, bones['m']['metacarpal'].prev_joint.y,
                                           bones['m']['metacarpal'].prev_joint.z,
                                           bones['m']['metacarpal'].prev_joint.pitch,
                                           bones['m']['metacarpal'].prev_joint.yaw,
                                           bones['m']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['m']['metacarpal'].next_joint.x, bones['m']['metacarpal'].next_joint.y,
                                           bones['m']['metacarpal'].next_joint.z,
                                           bones['m']['metacarpal'].next_joint.pitch,
                                           bones['m']['metacarpal'].next_joint.yaw,
                                           bones['m']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['m']['proximal'].center.x, bones['m']['proximal'].center.y,
                                       bones['m']['proximal'].center.z,
                                       bones['m']['proximal'].center.pitch, bones['m']['proximal'].center.yaw,
                                       bones['m']['proximal'].center.roll],
                            'direction': [bones['m']['proximal'].direction.x, bones['m']['proximal'].direction.y,
                                          bones['m']['proximal'].direction.z,
                                          bones['m']['proximal'].direction.pitch,
                                          bones['m']['proximal'].direction.yaw,
                                          bones['m']['proximal'].direction.roll],
                            'length': bones['m']['proximal'].length,
                            'width': bones['m']['proximal'].width,
                            'prev_joint': [bones['m']['proximal'].prev_joint.x, bones['m']['proximal'].prev_joint.y,
                                           bones['m']['proximal'].prev_joint.z,
                                           bones['m']['proximal'].prev_joint.pitch,
                                           bones['m']['proximal'].prev_joint.yaw,
                                           bones['m']['proximal'].prev_joint.roll],
                            'next_joint': [bones['m']['proximal'].next_joint.x, bones['m']['proximal'].next_joint.y,
                                           bones['m']['proximal'].next_joint.z,
                                           bones['m']['proximal'].next_joint.pitch,
                                           bones['m']['proximal'].next_joint.yaw,
                                           bones['m']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['m']['intermediate'].center.x, bones['m']['intermediate'].center.y,
                                       bones['m']['intermediate'].center.z,
                                       bones['m']['intermediate'].center.pitch, bones['m']['intermediate'].center.yaw,
                                       bones['m']['intermediate'].center.roll],
                            'direction': [bones['m']['intermediate'].direction.x,
                                          bones['m']['intermediate'].direction.y,
                                          bones['m']['intermediate'].direction.z,
                                          bones['m']['intermediate'].direction.pitch,
                                          bones['m']['intermediate'].direction.yaw,
                                          bones['m']['intermediate'].direction.roll],
                            'length': bones['m']['intermediate'].length,
                            'width': bones['m']['intermediate'].width,
                            'prev_joint': [bones['m']['intermediate'].prev_joint.x,
                                           bones['m']['intermediate'].prev_joint.y,
                                           bones['m']['intermediate'].prev_joint.z,
                                           bones['m']['intermediate'].prev_joint.pitch,
                                           bones['m']['intermediate'].prev_joint.yaw,
                                           bones['m']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['m']['intermediate'].next_joint.x,
                                           bones['m']['intermediate'].next_joint.y,
                                           bones['m']['intermediate'].next_joint.z,
                                           bones['m']['intermediate'].next_joint.pitch,
                                           bones['m']['intermediate'].next_joint.yaw,
                                           bones['m']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['m']['distal'].center.x, bones['m']['distal'].center.y,
                                       bones['m']['distal'].center.z,
                                       bones['m']['distal'].center.pitch, bones['m']['distal'].center.yaw,
                                       bones['m']['distal'].center.roll],
                            'direction': [bones['m']['distal'].direction.x, bones['m']['distal'].direction.y,
                                          bones['m']['distal'].direction.z,
                                          bones['m']['distal'].direction.pitch,
                                          bones['m']['distal'].direction.yaw,
                                          bones['m']['distal'].direction.roll],
                            'length': bones['m']['distal'].length,
                            'width': bones['m']['distal'].width,
                            'prev_joint': [bones['m']['distal'].prev_joint.x, bones['m']['distal'].prev_joint.y,
                                           bones['m']['distal'].prev_joint.z,
                                           bones['m']['distal'].prev_joint.pitch,
                                           bones['m']['distal'].prev_joint.yaw,
                                           bones['m']['distal'].prev_joint.roll],
                            'next_joint': [bones['m']['distal'].next_joint.x, bones['m']['distal'].next_joint.y,
                                           bones['m']['distal'].next_joint.z,
                                           bones['m']['distal'].next_joint.pitch,
                                           bones['m']['distal'].next_joint.yaw,
                                           bones['m']['distal'].next_joint.roll]
                        }
                    }
                },
                'ring': {
                    'id': fingers_list[3].id,
                    'length': fingers_list[3].length,
                    'width': fingers_list[3].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['r']['metacarpal'].center.x, bones['r']['metacarpal'].center.y,
                                       bones['r']['metacarpal'].center.z,
                                       bones['r']['metacarpal'].center.pitch, bones['r']['metacarpal'].center.yaw,
                                       bones['r']['metacarpal'].center.roll],
                            'direction': [bones['r']['metacarpal'].direction.x, bones['r']['metacarpal'].direction.y,
                                          bones['r']['metacarpal'].direction.z,
                                          bones['r']['metacarpal'].direction.pitch,
                                          bones['r']['metacarpal'].direction.yaw,
                                          bones['r']['metacarpal'].direction.roll],
                            'length': bones['r']['metacarpal'].length,
                            'width': bones['r']['metacarpal'].width,
                            'prev_joint': [bones['r']['metacarpal'].prev_joint.x, bones['r']['metacarpal'].prev_joint.y,
                                           bones['r']['metacarpal'].prev_joint.z,
                                           bones['r']['metacarpal'].prev_joint.pitch,
                                           bones['r']['metacarpal'].prev_joint.yaw,
                                           bones['r']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['r']['metacarpal'].next_joint.x, bones['r']['metacarpal'].next_joint.y,
                                           bones['r']['metacarpal'].next_joint.z,
                                           bones['r']['metacarpal'].next_joint.pitch,
                                           bones['r']['metacarpal'].next_joint.yaw,
                                           bones['r']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['r']['proximal'].center.x, bones['r']['proximal'].center.y,
                                       bones['r']['proximal'].center.z,
                                       bones['r']['proximal'].center.pitch, bones['r']['proximal'].center.yaw,
                                       bones['r']['proximal'].center.roll],
                            'direction': [bones['r']['proximal'].direction.x, bones['r']['proximal'].direction.y,
                                          bones['r']['proximal'].direction.z,
                                          bones['r']['proximal'].direction.pitch,
                                          bones['r']['proximal'].direction.yaw,
                                          bones['r']['proximal'].direction.roll],
                            'length': bones['r']['proximal'].length,
                            'width': bones['r']['proximal'].width,
                            'prev_joint': [bones['r']['proximal'].prev_joint.x, bones['r']['proximal'].prev_joint.y,
                                           bones['r']['proximal'].prev_joint.z,
                                           bones['r']['proximal'].prev_joint.pitch,
                                           bones['r']['proximal'].prev_joint.yaw,
                                           bones['r']['proximal'].prev_joint.roll],
                            'next_joint': [bones['r']['proximal'].next_joint.x, bones['r']['proximal'].next_joint.y,
                                           bones['r']['proximal'].next_joint.z,
                                           bones['r']['proximal'].next_joint.pitch,
                                           bones['r']['proximal'].next_joint.yaw,
                                           bones['r']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['r']['intermediate'].center.x, bones['r']['intermediate'].center.y,
                                       bones['r']['intermediate'].center.z,
                                       bones['r']['intermediate'].center.pitch, bones['r']['intermediate'].center.yaw,
                                       bones['r']['intermediate'].center.roll],
                            'direction': [bones['r']['intermediate'].direction.x,
                                          bones['r']['intermediate'].direction.y,
                                          bones['r']['intermediate'].direction.z,
                                          bones['r']['intermediate'].direction.pitch,
                                          bones['r']['intermediate'].direction.yaw,
                                          bones['r']['intermediate'].direction.roll],
                            'length': bones['r']['intermediate'].length,
                            'width': bones['r']['intermediate'].width,
                            'prev_joint': [bones['r']['intermediate'].prev_joint.x,
                                           bones['r']['intermediate'].prev_joint.y,
                                           bones['r']['intermediate'].prev_joint.z,
                                           bones['r']['intermediate'].prev_joint.pitch,
                                           bones['r']['intermediate'].prev_joint.yaw,
                                           bones['r']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['r']['intermediate'].next_joint.x,
                                           bones['r']['intermediate'].next_joint.y,
                                           bones['r']['intermediate'].next_joint.z,
                                           bones['r']['intermediate'].next_joint.pitch,
                                           bones['r']['intermediate'].next_joint.yaw,
                                           bones['r']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['r']['distal'].center.x, bones['r']['distal'].center.y,
                                       bones['r']['distal'].center.z,
                                       bones['r']['distal'].center.pitch, bones['r']['distal'].center.yaw,
                                       bones['r']['distal'].center.roll],
                            'direction': [bones['r']['distal'].direction.x, bones['r']['distal'].direction.y,
                                          bones['r']['distal'].direction.z,
                                          bones['r']['distal'].direction.pitch,
                                          bones['r']['distal'].direction.yaw,
                                          bones['r']['distal'].direction.roll],
                            'length': bones['r']['distal'].length,
                            'width': bones['r']['distal'].width,
                            'prev_joint': [bones['r']['distal'].prev_joint.x, bones['r']['distal'].prev_joint.y,
                                           bones['r']['distal'].prev_joint.z,
                                           bones['r']['distal'].prev_joint.pitch,
                                           bones['r']['distal'].prev_joint.yaw,
                                           bones['r']['distal'].prev_joint.roll],
                            'next_joint': [bones['r']['distal'].next_joint.x, bones['r']['distal'].next_joint.y,
                                           bones['r']['distal'].next_joint.z,
                                           bones['r']['distal'].next_joint.pitch,
                                           bones['r']['distal'].next_joint.yaw,
                                           bones['r']['distal'].next_joint.roll]
                        }
                    }
                },
                'pinky': {
                    'id': fingers_list[4].id,
                    'length': fingers_list[4].length,
                    'width': fingers_list[4].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['p']['metacarpal'].center.x, bones['p']['metacarpal'].center.y,
                                       bones['p']['metacarpal'].center.z,
                                       bones['p']['metacarpal'].center.pitch, bones['p']['metacarpal'].center.yaw,
                                       bones['p']['metacarpal'].center.roll],
                            'direction': [bones['p']['metacarpal'].direction.x, bones['p']['metacarpal'].direction.y,
                                          bones['p']['metacarpal'].direction.z,
                                          bones['p']['metacarpal'].direction.pitch,
                                          bones['p']['metacarpal'].direction.yaw,
                                          bones['p']['metacarpal'].direction.roll],
                            'length': bones['p']['metacarpal'].length,
                            'width': bones['p']['metacarpal'].width,
                            'prev_joint': [bones['p']['metacarpal'].prev_joint.x, bones['p']['metacarpal'].prev_joint.y,
                                           bones['p']['metacarpal'].prev_joint.z,
                                           bones['p']['metacarpal'].prev_joint.pitch,
                                           bones['p']['metacarpal'].prev_joint.yaw,
                                           bones['p']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['p']['metacarpal'].next_joint.x, bones['p']['metacarpal'].next_joint.y,
                                           bones['p']['metacarpal'].next_joint.z,
                                           bones['p']['metacarpal'].next_joint.pitch,
                                           bones['p']['metacarpal'].next_joint.yaw,
                                           bones['p']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['p']['proximal'].center.x, bones['p']['proximal'].center.y,
                                       bones['p']['proximal'].center.z,
                                       bones['p']['proximal'].center.pitch, bones['p']['proximal'].center.yaw,
                                       bones['p']['proximal'].center.roll],
                            'direction': [bones['p']['proximal'].direction.x, bones['p']['proximal'].direction.y,
                                          bones['p']['proximal'].direction.z,
                                          bones['p']['proximal'].direction.pitch,
                                          bones['p']['proximal'].direction.yaw,
                                          bones['p']['proximal'].direction.roll],
                            'length': bones['p']['proximal'].length,
                            'width': bones['p']['proximal'].width,
                            'prev_joint': [bones['p']['proximal'].prev_joint.x, bones['p']['proximal'].prev_joint.y,
                                           bones['p']['proximal'].prev_joint.z,
                                           bones['p']['proximal'].prev_joint.pitch,
                                           bones['p']['proximal'].prev_joint.yaw,
                                           bones['p']['proximal'].prev_joint.roll],
                            'next_joint': [bones['p']['proximal'].next_joint.x, bones['p']['proximal'].next_joint.y,
                                           bones['p']['proximal'].next_joint.z,
                                           bones['p']['proximal'].next_joint.pitch,
                                           bones['p']['proximal'].next_joint.yaw,
                                           bones['p']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['p']['intermediate'].center.x, bones['p']['intermediate'].center.y,
                                       bones['p']['intermediate'].center.z,
                                       bones['p']['intermediate'].center.pitch, bones['p']['intermediate'].center.yaw,
                                       bones['p']['intermediate'].center.roll],
                            'direction': [bones['p']['intermediate'].direction.x,
                                          bones['p']['intermediate'].direction.y,
                                          bones['p']['intermediate'].direction.z,
                                          bones['p']['intermediate'].direction.pitch,
                                          bones['p']['intermediate'].direction.yaw,
                                          bones['p']['intermediate'].direction.roll],
                            'length': bones['p']['intermediate'].length,
                            'width': bones['p']['intermediate'].width,
                            'prev_joint': [bones['p']['intermediate'].prev_joint.x,
                                           bones['p']['intermediate'].prev_joint.y,
                                           bones['p']['intermediate'].prev_joint.z,
                                           bones['p']['intermediate'].prev_joint.pitch,
                                           bones['p']['intermediate'].prev_joint.yaw,
                                           bones['p']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['p']['intermediate'].next_joint.x,
                                           bones['p']['intermediate'].next_joint.y,
                                           bones['p']['intermediate'].next_joint.z,
                                           bones['p']['intermediate'].next_joint.pitch,
                                           bones['p']['intermediate'].next_joint.yaw,
                                           bones['p']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['p']['distal'].center.x, bones['p']['distal'].center.y,
                                       bones['p']['distal'].center.z,
                                       bones['p']['distal'].center.pitch, bones['p']['distal'].center.yaw,
                                       bones['p']['distal'].center.roll],
                            'direction': [bones['p']['distal'].direction.x, bones['p']['distal'].direction.y,
                                          bones['p']['distal'].direction.z,
                                          bones['p']['distal'].direction.pitch,
                                          bones['p']['distal'].direction.yaw,
                                          bones['p']['distal'].direction.roll],
                            'length': bones['p']['distal'].length,
                            'width': bones['p']['distal'].width,
                            'prev_joint': [bones['p']['distal'].prev_joint.x, bones['p']['distal'].prev_joint.y,
                                           bones['p']['distal'].prev_joint.z,
                                           bones['p']['distal'].prev_joint.pitch,
                                           bones['p']['distal'].prev_joint.yaw,
                                           bones['p']['distal'].prev_joint.roll],
                            'next_joint': [bones['p']['distal'].next_joint.x, bones['p']['distal'].next_joint.y,
                                           bones['p']['distal'].next_joint.z,
                                           bones['p']['distal'].next_joint.pitch,
                                           bones['p']['distal'].next_joint.yaw,
                                           bones['p']['distal'].next_joint.roll]
                        }
                    }
                }
            },

            'pointables': {
                'p_0':{
                    'tip_position': [pointables_list[0].tip_position.x, pointables_list[0].tip_position.y, pointables_list[0].tip_position.z,
                                     pointables_list[0].tip_position.pitch, pointables_list[0].tip_position.yaw, pointables_list[0].tip_position.roll],
                    'tip_velocity': [pointables_list[0].tip_velocity.x, pointables_list[0].tip_velocity.y, pointables_list[0].tip_velocity.z,
                                     pointables_list[0].tip_velocity.pitch, pointables_list[0].tip_velocity.yaw, pointables_list[0].tip_velocity.roll],
                    'direction': [pointables_list[0].direction.x, pointables_list[0].direction.y, pointables_list[0].direction.z,
                                     pointables_list[0].direction.pitch, pointables_list[0].direction.yaw, pointables_list[0].direction.roll],
                    'width': pointables_list[0].width,
                    'length': pointables_list[0].length,
                    'is_extended': pointables_list[0].is_extended
                },
                'p_1':{
                    'tip_position': [pointables_list[1].tip_position.x, pointables_list[1].tip_position.y,
                                     pointables_list[1].tip_position.z,
                                     pointables_list[1].tip_position.pitch, pointables_list[1].tip_position.yaw,
                                     pointables_list[1].tip_position.roll],
                    'tip_velocity': [pointables_list[1].tip_velocity.x, pointables_list[1].tip_velocity.y,
                                     pointables_list[1].tip_velocity.z,
                                     pointables_list[1].tip_velocity.pitch, pointables_list[1].tip_velocity.yaw,
                                     pointables_list[1].tip_velocity.roll],
                    'direction': [pointables_list[1].direction.x, pointables_list[1].direction.y,
                                  pointables_list[1].direction.z,
                                  pointables_list[1].direction.pitch, pointables_list[1].direction.yaw,
                                  pointables_list[1].direction.roll],
                    'width': pointables_list[1].width,
                    'length': pointables_list[1].length,
                    'is_extended': pointables_list[1].is_extended
                },
                'p_2': {
                    'tip_position': [pointables_list[2].tip_position.x, pointables_list[2].tip_position.y,
                                     pointables_list[2].tip_position.z,
                                     pointables_list[2].tip_position.pitch, pointables_list[2].tip_position.yaw,
                                     pointables_list[2].tip_position.roll],
                    'tip_velocity': [pointables_list[2].tip_velocity.x, pointables_list[2].tip_velocity.y,
                                     pointables_list[2].tip_velocity.z,
                                     pointables_list[2].tip_velocity.pitch, pointables_list[2].tip_velocity.yaw,
                                     pointables_list[2].tip_velocity.roll],
                    'direction': [pointables_list[2].direction.x, pointables_list[2].direction.y,
                                  pointables_list[2].direction.z,
                                  pointables_list[2].direction.pitch, pointables_list[2].direction.yaw,
                                  pointables_list[2].direction.roll],
                    'width': pointables_list[2].width,
                    'length': pointables_list[2].length,
                    'is_extended': pointables_list[2].is_extended
                },
                'p_3': {
                    'tip_position': [pointables_list[3].tip_position.x, pointables_list[3].tip_position.y,
                                     pointables_list[3].tip_position.z,
                                     pointables_list[3].tip_position.pitch, pointables_list[3].tip_position.yaw,
                                     pointables_list[3].tip_position.roll],
                    'tip_velocity': [pointables_list[3].tip_velocity.x, pointables_list[3].tip_velocity.y,
                                     pointables_list[3].tip_velocity.z,
                                     pointables_list[3].tip_velocity.pitch, pointables_list[3].tip_velocity.yaw,
                                     pointables_list[3].tip_velocity.roll],
                    'direction': [pointables_list[3].direction.x, pointables_list[3].direction.y,
                                  pointables_list[3].direction.z,
                                  pointables_list[3].direction.pitch, pointables_list[3].direction.yaw,
                                  pointables_list[3].direction.roll],
                    'width': pointables_list[3].width,
                    'length': pointables_list[3].length,
                    'is_extended': pointables_list[3].is_extended
                },
                'p_4': {
                    'tip_position': [pointables_list[4].tip_position.x, pointables_list[4].tip_position.y,
                                     pointables_list[4].tip_position.z,
                                     pointables_list[4].tip_position.pitch, pointables_list[4].tip_position.yaw,
                                     pointables_list[4].tip_position.roll],
                    'tip_velocity': [pointables_list[4].tip_velocity.x, pointables_list[4].tip_velocity.y,
                                     pointables_list[4].tip_velocity.z,
                                     pointables_list[4].tip_velocity.pitch, pointables_list[4].tip_velocity.yaw,
                                     pointables_list[4].tip_velocity.roll],
                    'direction': [pointables_list[4].direction.x, pointables_list[4].direction.y,
                                  pointables_list[4].direction.z,
                                  pointables_list[4].direction.pitch, pointables_list[4].direction.yaw,
                                  pointables_list[4].direction.roll],
                    'width': pointables_list[4].width,
                    'length': pointables_list[4].length,
                    'is_extended': pointables_list[4].is_extended
                }

            },
            'arm': {
                'width': h.arm.width,
                'direction': [h.arm.direction.x, h.arm.direction.y, h.arm.direction.z, h.arm.direction.pitch, h.arm.direction.yaw, h.arm.direction.roll],
                'wrist_position': [h.arm.wrist_position.x, h.arm.wrist_position.y, h.arm.wrist_position.z, h.arm.wrist_position.pitch, h.arm.wrist_position.yaw, h.arm.wrist_position.roll],
                'elbow_position': [h.arm.elbow_position.x, h.arm.elbow_position.y, h.arm.elbow_position.z, h.arm.elbow_position.pitch, h.arm.elbow_position.yaw, h.arm.elbow_position.roll],
            },
        }
    }
    return j_frame



