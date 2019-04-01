import utils
import cv2
import numpy as np
import queue
import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
import argparse
import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F
from models import CrossConvNet
import time

asd = 4
#
# src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# # Windows and Linux
# arch_dir = './leap_lib'
#
# sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
#
# # from leap_lib import Leap

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--n_frames', type=int, default=40)
parser.add_argument('--ir_model_path', type=str, default="weights/best_val_acc_checkpoint_DenseNet161P_depth_ir_40f_ep_49.pth.tar")
parser.add_argument('--depth_model_path', type=str, default="weights/best_val_acc_checkpoint_DenseNet161P_depth_z_40f_ep_6.pth.tar")
parser.add_argument('--pred_th', type=float, default=0.8)



gestures = ['FIST', 'PINCH', 'FLIP-OVER', 'TELEPHONE', 'RIGHT SWIPE', 'LEFT SWIPE', 'TOP-DOWN SWIPE', 'BOTTOM-UP SWIPE',
            'THUMB', 'INDEX', 'CLOCKWISE ROTATION', 'COUNTERCLOCKWISE ROTATION']


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        z_values = []
        gray_values = []
        for i in range(data.getNumPoints()):
            z_values.append(data.getZ(i))
            gray_values.append(data.getGrayValue(i))

        z_array = np.array(z_values)
        gray_array = np.array(gray_values)

        z_p = z_array.reshape(-1, data.width)
        gray_p = gray_array.reshape(-1, data.width)

        self.queue.put((z_p, gray_p))


def set_up(controller, rgb_cam=0):

    print("waiting for maps initialization...")
    while True:

        frame = controller.frame()
        image_l = frame.images[0]
        image_r = frame.images[1]

        if image_l.is_valid and image_r.is_valid:

            left_coordinates, left_coefficients = utils.convert_distortion_maps(image_l)
            right_coordinates, right_coefficients = utils.convert_distortion_maps(image_r)
            maps_initialized = True
            print('maps initialized')

            break
        else:
            print('\rinvalid leap motion frame', end="")

    # # initialize video capture
    # while True:
    #     cap = cv2.VideoCapture(rgb_cam)
    #     print(cap)
    #     if cap:
    #         # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
    #         # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)
    #         # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         break
    #     else:
    #         print("\rerror rgb cam", end="")
    #
    # print("ready to go")
    # return  cap


def run(controller, cam, model, device, mean_depth, std_depth, mean_ir, std_ir):

    # inizializzazione picoflexx
    # q = queue.Queue()
    # listener = MyListener(q)
    # cam.registerDataListener(listener)

    # setting up cameras
    # cap = set_up(controller, rgb_cam=0)

    while True:


        print("press E to start")
        utils.draw_ui(text="press E to start")

        k = cv2.waitKey()
        if k == ord('e'):


            # RUNNING
            # if not cap:
            #     print("error rgb cam")
            #     exit(-1)
            error_queue = False
            stop = False

            # while True:
            #     try:
            #         q = queue.Queue()
            #         listener = MyListener(q)
            #         cam.registerDataListener(listener)
            #         cam.startCapture()
            #         print("picoflex ready")
            #         break
            #     except RuntimeError:
            #         print("error connection picoflex")

            buffer_depth, buffer_ir = [], []
            counter = 0
            # softmax = nn.Softmax()
            pause = False
            detected = None
            while True:
                # time.sleep(0.1)
                if len(buffer_depth) == 0:
                    while True:
                        try:
                            q = queue.Queue()
                            listener = MyListener(q)
                            cam.registerDataListener(listener)
                            cam.startCapture()
                            print("picoflex ready")
                            break
                        except RuntimeError:
                            print("error connection picoflex")
                    print("start")
                if cv2.waitKey(1) == ord('s') or pause or error_queue:
                    stop = True
                    pause = False
                    break
                # print(frame_counter)

                if error_queue:
                    print(error_queue)
                    break
                # print("\rRunning...", end="")
                # utils.draw_ui(text="Running... press S to stop")

                # PICOFLEXX
                # imgs == (z, ir)
                ret_pico, imgs = utils.get_images_from_picoflexx(q)
                # print("ret_pico, z, ir", ret_pico, imgs[0], imgs[1])
                if not ret_pico:
                    print("pico image not valid")
                    error_queue = True
                    # break
                # show images
                else:

                    counter += 1
                    print('buffer: {}'.format(len(buffer_depth)))
                    depth_x = imgs[0]
                    ir_x = imgs[1]

                    # cv2.imshow('img_ir', cv2.resize(ir_x, (0, 0), fx=1.5, fy=1.5))
                    # cv2.imshow('img_depth', cv2.resize((depth_x * 255).astype(np.uint8), (0, 0), fx=1.5, fy=1.5))

                    utils.draw_demo_ui("", img0=cv2.resize(imgs[1], (0, 0), fx=1.5, fy=1.5),
                                       img1=cv2.resize((imgs[0] * 65535).astype(np.uint16), (0, 0), fx=1.5, fy=1.5),
                                       text="RUNNING..." if detected is None or counter > 20 else detected)

                    # inserisco in buffer
                    if len(buffer_depth) < args.n_frames:
                        # resize
                        depth_x = np.expand_dims(cv2.resize(depth_x, (224, 224)), axis=2)
                        ir_x = np.expand_dims(cv2.resize(ir_x, (224, 224)), axis=2)

                        buffer_depth.append(depth_x)
                        buffer_ir.append(ir_x)

                    elif len(buffer_depth) >= args.n_frames:
                        # resize
                        depth_x = np.expand_dims(cv2.resize(depth_x, (224, 224)), axis=2)
                        ir_x = np.expand_dims(cv2.resize(ir_x, (224, 224)), axis=2)

                        buffer_depth.pop(0)
                        buffer_depth.append(depth_x)

                        buffer_ir.pop(0)
                        buffer_ir.append(ir_x)

                    if len(buffer_depth) == args.n_frames:
                        # prediction

                        # creo clip
                        clip_depth = np.concatenate(buffer_depth, axis=2)
                        clip_depth = np.float32(clip_depth.transpose([2, 0, 1]))

                        clip_ir = np.concatenate(buffer_ir, axis=2)
                        clip_ir = np.float32(clip_ir.transpose([2, 0, 1]))
                        # pre processing



                        # normalization
                        clip_depth = (clip_depth - mean_depth)/std_depth
                        clip_ir = (clip_ir - mean_ir)/std_ir

                        # converto in tensori
                        clip_depth = torch.tensor(clip_depth)
                        clip_ir = torch.tensor(clip_ir)

                        # passo in gpu
                        clip_depth = clip_depth.to(device)
                        clip_ir = clip_ir.to(device)

                        # predizione
                        out = model(clip_depth.unsqueeze(dim=0) #, clip_ir.unsqueeze(dim=0))
                                    )
                        # print("out: {}, predicted: {}".format(out, torch.max(out, 1)))
                        out = F.softmax(out, dim=1)
                        out = torch.max(out, 1)
                        if out[0] >= args.pred_th:
                            print("detected ", out[0].item(), out[1].item())
                        # print("softmax: {}", out)
                            detected = gestures[out[1].item()]

                            buffer_depth = []
                            buffer_ir = []
                            counter = 0
                            # pause = True
                            cam.stopCapture()
                            print("stop")
                            # utils.draw_ui("{} detected".format(gestures[out[1].item()]), position=(255, 255))
                            utils.draw_demo_ui("", img0=cv2.resize(imgs[1], (0, 0), fx=1.5, fy=1.5),
                                               img1=cv2.resize((imgs[0] * 65535).astype(np.uint16), (0, 0), fx=1.5,
                                                               fy=1.5),
                                               text=detected)


                        else:
                            print("not_recognized")
                            # print("not detected: ", out[0].item(), out[1].item())
                            # buffer_depth = []
                            # buffer_ir = []
                            # counter = 0
                            # # time.sleep(10)
                            # pause = True
                            pass
                            #keep recording



args = parser.parse_args()


def main():
    # PICOFLEXX

    parser1 = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser1)

    # parser1.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser1.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()

    cam.setUseCase("MODE_5_45FPS_500")
    # print_camera_info(cam)
    # print("isConnected", cam.isConnected())
    # print("getFrameRate", cam.getFrameRate())

    # # LEAP MOTION
    # controller = Leap.Controller()
    # controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)


    # CARICO RETI

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(1994)
        torch.backends.cudnn.deterministic = True

    # DEPTH
    model_depth = models.densenet161(pretrained=True)
    # for params in model_depth.parameters():
    #     params.requires_grad = False
    model_depth.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                       stride=(2, 2), padding=(3, 3))
    model_depth.classifier = nn.Linear(in_features=2208, out_features=12, bias=True)

    # carico i pesi

    model_depth.load_state_dict(torch.load(args.depth_model_path, map_location=device)['state_dict'])


    model_depth = model_depth.to(device)
    model_depth.eval()




    # IR
    model_ir = models.densenet161(pretrained=True)
    # for params in model_ir.parameters():
    #     params.requires_grad = False
    model_ir.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                    stride=(2, 2), padding=(3, 3))
    model_ir.classifier = nn.Linear(in_features=2208, out_features=12, bias=True)

    model_ir.load_state_dict(torch.load(args.ir_model_path, map_location=device)['state_dict'])

    model_ir = model_ir.to(device)
    model_ir.eval()



    # create CROSS CONVNET

    # model = CrossConvNet(n_classes=12, depth_model=model_depth, ir_model=None, rgb_model=None,
    #                             mode='depth_ir', cross_mode='avg').to(device)
    # model.eval()


    # carico media e std

    npzfile = np.load("weights/mean_std_depth_z.npz")
    mean_depth = npzfile['arr_0']
    std_depth = npzfile['arr_1']
    print('mean, std depth_z loaded.')

    npzfile = np.load("weights/mean_std_depth_ir.npz")
    mean_ir = npzfile['arr_0']
    std_ir = npzfile['arr_1']
    print('mean, std depth_ir loaded.')

    run(controller=None, cam=cam, model=model_depth, device=device, mean_depth=mean_depth, std_depth=std_depth,
        mean_ir=mean_ir, std_ir=std_ir)


if __name__ == '__main__':
    main()

