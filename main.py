import sys
import os
import time
from argparse import Namespace

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
from test import copyStateDict, test_net

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize
    args = Namespace()
    args.__dict__.update({'trained_model':'weights/craft_mlt_25k.pth',
                        'text_threshold':0.7,
                        'low_text':0.4,
                        'link_threshold':0.4,
                        'cuda': sys.platform == 'win32',
                        'canvas_size':1280,
                        'mag_ratio':1.5,
                        'poly':False,
                        'show_time':False,
                        'test_folder':'/data/',
                        'refine':False,
                        'refiner_model':'weights/craft_refiner_CTW1500.pth',
                        })
    
    image_list, _, _ = file_utils.get_files(args.test_folder)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
        
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
