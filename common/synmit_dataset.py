from __future__ import absolute_import, division

import numpy as np
import copy
import os
import re
import torch
import pickle
from pathlib import *
import json_tricks as json
import xml.etree.cElementTree as ET
from common.camera import normalize_screen_coordinates_multiview, image_coordinates_multiview

class SynDataset17():
    def __init__(self, root, image_set):
        self.subset = image_set
        self.root = root
        self.db = []
        self.image_shape = [1600, 1200]
        self.view_count = 8
        self.joint_count = 17
        self.grouping = self._get_db()
        self.normalize2d()
        self.group_size = len(self.grouping)
        self._edge = [(0,1), (1, 2), (2, 6), (3, 6), (3, 4), (5, 4), (6, 7),
                     (7, 8), (8, 16), (16,9), (8, 12), (8, 13), (12, 11),
                     (11, 10), (13, 14), (14, 15)]
        self._nodes_group = [[0,1], [2,3], [4,5], [6,7], [10,11], [14,15], [12,13], [9,16], [8]]
    def __len__(self):
        return self.group_size

    def edge(self):
        return self._edge

    def jointcount(self):
        return self.joint_count

    def nodesgroup(self):
        return self._nodes_group

    def cam2projm(self, camera):
        K = camera['K']
        R = camera['R']
        T = camera['T']
        RT = np.hstack([R,T])
        return np.matmul(K,RT)

    def __getitem__(self, idx):
        frame = copy.deepcopy(self.grouping[idx])

        cameras = frame['camera_proj_mat']
        cameras = torch.from_numpy(np.array(cameras)).float()  # 8*3*4

        target_3d = frame['joints_3d']
        target_3d = torch.tensor(target_3d).float()

        networkpred_path = frame['network_pred_path'] + "_ltn17"

        input = frame['backbone_joints2d_normalized']  # 8*17*3
        input = torch.from_numpy(np.array(input)).float()

        input_fortri = frame['backbone_joints2d']
        input_fortri = torch.from_numpy(np.array(input_fortri)).float()

        target_score = frame['target_score']
        target_score = torch.from_numpy(np.array(target_score)).float()  # J,V

        target_dict = {
            "target_score": target_score,
            "target_3d": target_3d,
            "camera": cameras,
            "2d_ori": input_fortri,
            # "network_pred_path": networkpred_path
        }

        return target_score, input, target_dict

    def abspath2remotepath(self, abspath):
        search_str = "animation_mit"
        winpath = PureWindowsPath(abspath[abspath.index(search_str) + len(search_str) + 1:])
        root_path = Path(self.root)
        real_path = root_path / winpath
        return str(real_path)

    def parsecalibration(self, calibfolder):

        extrinsics = os.path.join(calibfolder, 'extrinsics.xml')
        intrinsic = os.path.join(calibfolder, 'intrinsic.xml')

        tree = ET.ElementTree(file=extrinsics)
        root = tree.getroot()
        # get R
        R_node = root[0]
        text = R_node[3].text
        R_data = re.split('[\s]\s*', text)
        R_data.remove('')
        R = list(map(eval, R_data))
        R = np.array(R)
        R = R.reshape(3, 3)
        # print(R)

        # get T
        T_node = root[1]
        text = T_node[3].text
        T_data = re.split('[\s]\s*', text)
        T_data.remove('')
        T = list(map(eval, T_data))
        T = np.array(T)
        T = T.reshape(3, 1)

        # load intrinsic
        tree = ET.ElementTree(file=intrinsic)
        root = tree.getroot()
        # get K
        date = root[0]
        if date.tag == "date":
            M_node = 2
        else:
            M_node = 0

        K_node = root[M_node]
        # K_node = root[2]
        text = K_node[3].text
        K_data = re.split('[\s]\s*', text)
        K_data.remove('')
        K = list(map(eval, K_data))
        K = np.array(K)
        K = K.reshape(3, 3)

        return K, R, T

    def loadransacidx(self, ransacidxfile, backbone2d_ws):
        fopen = open(ransacidxfile, 'r')
        lines = fopen.readlines()
        fopen.close()
        for i in range(self.joint_count):
            weights = np.zeros(self.view_count)
            selectid = lines[i].split(" ")
            for j in selectid:
                if j != "\n":
                    weights[int(j)-1] = 1
            outlier_id = weights == 0
            backbone2d_ws[outlier_id, i, 2] = self.ransac_min_weight

        return backbone2d_ws

    def openpose25toh36m17(selfs, joints_3d):
        mapid = [11,10,9,12,13,14,8,1,1,16,4,3,2,5,6,7,0]
        joints_3d_17 = joints_3d[mapid]
        joints_3d_17[7] = (joints_3d[1] + joints_3d[8])/2
        joints_3d_17[9] = (joints_3d[15] + joints_3d[16]) / 2
        return joints_3d_17

    def normalize2d(self):
        #normalize input
        for i in range(len(self.grouping)):
            joints2d = copy.deepcopy(self.grouping[i]['backbone_joints2d'])
            joints2d[:, :, 0:2] = normalize_screen_coordinates_multiview(joints2d[:, :, 0:2],
                                                                        self.image_shape[1],
                                                                        self.image_shape[0])
            self.grouping[i]['backbone_joints2d_normalized'] = joints2d

    def _get_db(self):
        gt_db = []
        pkl_name = os.path.join(self.root, 'list', 'pkl', self.subset + 'ltn17.pkl')
        ########################### load from pkl
        if os.path.exists(pkl_name):
            print("Loading dataset from: " + pkl_name)
            with open(pkl_name,'rb') as f:
                gt_db = pickle.load(f)
            return gt_db
        else:
            if not os.path.exists(os.path.join(self.root, 'list','pkl')):
                os.mkdir(os.path.join(self.root, 'list', 'pkl'))

        ############################ load from folder
        file_name = os.path.join(self.root, 'list', self.subset + 'lst.txt')
        json_name = os.path.join(self.root, 'list', 'datalist.json')
        with open(json_name, 'r') as load_f:
            load_dict = json.load(load_f)
        with open(file_name) as anno_file:
            anno = anno_file.readlines()

        print("Total frames: " + str(len(anno)))

        for items in anno: # per frame
            if (len(gt_db)) % 100 == 0:
                print("Loading " + str(len(gt_db)) + " Frames")
            frame_folder = items.split(";")[0]
            frame_folder_abs = self.abspath2remotepath(frame_folder)
            network_pred_path = os.path.join(frame_folder_abs, 'openpose', 'network_skeleton_body')

            dict_id = int(items.split(";")[1])
            framefiles = os.listdir(frame_folder_abs)
            imgname_pattern = load_dict[dict_id]['image_name_format']

            ### load 2d_backbone as input
            input2dfile = os.path.join(frame_folder_abs, "openpose", "just_input_fix-ltn17.txt")
            joints_2d_backbone_wscore_all = np.loadtxt(input2dfile)
            joints_2d_backbone_wscore_all = np.delete(joints_2d_backbone_wscore_all, 2, axis=1)
            if joints_2d_backbone_wscore_all.shape[0] != self.joint_count * self.view_count:
                print(input2dfile)

            # print(openposefile)
            joints_2d_backbone_wscore = joints_2d_backbone_wscore_all.reshape(self.joint_count, self.view_count, -1)
            joints_2d_backbone_wscore = joints_2d_backbone_wscore.transpose((1, 0, 2)) # view, joint, 3

            ## load 3d ground truth
            joints_3d_file = os.path.join(frame_folder_abs, load_dict[dict_id]['gt_skeleton'],
                                          'skeleton_body', 'skeleton.txt')
            joints_3d = np.loadtxt(joints_3d_file)
            joints_3d = self.openpose25toh36m17(joints_3d)

            ## load 3d ransac
            if self.with_ransac:
                joints_3d_file_ransac = os.path.join(frame_folder_abs, 'openpose', 'gcn_skeleton_body', 'skeleton.txt')
                joints_3d_ransac = np.loadtxt(joints_3d_file_ransac)
                joints_3d_file_ransac_selected = os.path.join(frame_folder_abs, 'openpose', 'gcn_skeleton_body',
                                                              'selectedViews.txt')
                ## fix input score
                joints_2d_backbone_wscore = self.loadransacidx(joints_3d_file_ransac_selected, joints_2d_backbone_wscore)
            else:
                joints_3d_ransac = np.zeros([self.joint_count,3])
            imgnames = []
            camera_ps = []
            joints_2ds = []
            for imgfile in framefiles:
                if re.match(imgname_pattern, imgfile) is not None:
                    imgname = os.path.join(frame_folder_abs, imgfile)
                    imgnames.append(imgname)

                    camname_len = int(imgname_pattern[9])
                    camname = imgname[imgname.index("cam"):imgname.index("cam") + 3 + camname_len] + "_000000"
                    calibration_folder = self.abspath2remotepath(load_dict[dict_id]['calibration'])
                    calibration_folder_camera = os.path.join(calibration_folder, camname)

                    ## load camera
                    K, R, T = self.parsecalibration(calibration_folder_camera)
                    camera = {'K': K, 'R': R, 'T': T}
                    camera_p = self.cam2projm(camera)
                    camera_ps.append(camera_p)

                    ## project to get 2d ground truth
                    joints_2d, joints_2d_vis = self.project3d(K, R, T, joints_3d, self.image_shape[0], self.image_shape[1])
                    joints_2d[joints_2d_vis==0, :] = 1e5
                    joints_2ds.append(joints_2d)
                    #
                    # plot skeleton
                    # image = cv2.imread(imgname)
                    # for j in joints_2d: # gt projection
                    #     cv2.circle(image, (int(j[0]), int(j[1])), 5, (0,255,0), thickness=-1)
                    #
                    # for j in joints_2d_backbone_wscore[len(imgnames)-1,:,:]: # prediction
                    #     cv2.circle(image, (int(j[0]), int(j[1])), 5, (255, 0, 0), thickness=-1)
                    # plt.figure()
                    # plt.title(imgname)
                    # plt.imshow(image)
                    # plt.show()


            joints_2ds = np.array(joints_2ds)
            target_score = np.linalg.norm(joints_2ds - joints_2d_backbone_wscore[:, :, 0:2], axis=2)
            target_score = 1 / target_score
            # target_score = np.exp(-0.5 * (target_score / self.sigma) ** 2)
            target_score = target_score.transpose((1, 0))  # J,V

            # svd3d = self.batch_triangulation(torch.from_numpy(joints_2ds.transpose((2,1,0))), torch.from_numpy(np.array(camera_ps)));

            gt_db.append(
                {
                    'target_score':target_score,
                    'source': self.cfg.DATASET.DATASET_TYPE,
                    'backbone_joints2d': joints_2d_backbone_wscore,
                    'joints_3d': joints_3d,
                    'joints_3d_ransac': joints_3d_ransac,
                    'network_pred_path': network_pred_path,
                    'image': imgnames,
                    'joints_2d': joints_2ds,
                    'camera_proj_mat': camera_ps,
                }
            )

        with open(pkl_name, 'wb') as f:
            pickle.dump(gt_db, f, pickle.HIGHEST_PROTOCOL)

        print("Init Dataset Done!!")
        return gt_db

    def project3d(self, K, R, T, joints_3d, rows, cols):
        joint_count = joints_3d.shape[0]

        RT = np.c_[R, T]
        joints_3d_hom = np.c_[joints_3d, np.ones(joint_count)]

        joints_3d_w2c = np.matmul(RT, joints_3d_hom.transpose())
        joint_2d_wh = np.matmul(K, joints_3d_w2c)
        joint_2d_w = joint_2d_wh / joint_2d_wh[2]
        joint_2d_w = joint_2d_w[0:2,:].transpose()

        joints_2d_vis_x = np.array(joint_2d_w[:,0] >= 0) & np.array(joint_2d_w[:,0] < cols)
        joints_2d_vis_y = np.array(joint_2d_w[:,1] >= 0) & np.array(joint_2d_w[:,1] < rows)
        joints_2d_vis = (joints_2d_vis_x.flatten() & joints_2d_vis_y.flatten()).astype(int).flatten()

        return joint_2d_w, joints_2d_vis