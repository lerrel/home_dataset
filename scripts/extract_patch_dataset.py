'''
Code that extract patch dataset from the home grasping data collected in http://papers.nips.cc/paper/8123-robot-learning-in-homes-improving-generalization-and-reducing-dataset-bias.pdf

Example run:
    python extract_patch_dataset.py --home_dataset_path ~/home_dataset/grasping_data --patch_dataset_path '/tmp/' --train_fraction 0.8 --display 1 --msec 1000
'''
import cloudpickle as pkl
import os
from os.path import join, getsize
from IPython import embed
import cv2
import numpy as np
from copy import deepcopy as copy
import tempfile
import shutil
import csv
import scipy
from scipy import misc
import random
import string
import argparse

HOME_DIR = os.environ['HOME']
DATA_ROOT_DIR = os.path.join(HOME_DIR,'home_dataset/grasping_data_small')
GRASP_SUBNAME = 'grasp_'
DATA_FILE = 'data.p'
ESSENTIAL_KEYS = ['grasp_x_img', 'grasp_y_img', 'valid_grasp', 'grasp_theta', 'grasp_result']
VALID_KEY = 'valid_grasp'
SUCCESS_KEY = 'grasp_result'
IMAGE_FILE = 'color.jpg'
DEPTH_FILE = 'depth.jpg'
PATCH_SIZE = 150

## Balances the dataset such that each grasp environment has atleast MIN_SUCCESS_RATE rate of grasp success
MIN_SUCCESS_RATE = 0.25
# MIN_SUCCESS_RATE = 0.0 

class DataHandler(object):
    def __init__(self, train_root_dir, val_root_dir=None, train_rf=[''], train_ef=[''], val_rf=[''], val_ef=['']):
        self.train_grasp_dirs = np.array(self.parse_grasp_data(train_root_dir, train_rf, train_ef))
        self.train_dataset = self.randomize_dataset(self.train_grasp_dirs)
        if val_root_dir != None:
            self.val_grasp_dirs = np.array(self.parse_grasp_data(val_root_dir, val_rf, val_ef))
            self.val_dataset = self.randomize_dataset(self.val_grasp_dirs)

    def write_train_val(self, data_dir, **kwargs):
        self.write_dataset(self.train_dataset, os.path.join(data_dir, 'Train'), **kwargs)
        self.write_dataset(self.val_dataset, os.path.join(data_dir, 'Validation'), **kwargs)

    def random_val_split(self, dataset, train_fraction=0.8):
        dataset = copy(dataset)
        l = len(dataset)
        nt = int(l*train_fraction)
        perms = np.random.permutation(l)
        self.train_dataset = dataset[perms[:nt]]
        self.val_dataset = dataset[perms[nt:]]

    def write_dataset(self, dataset, data_dir, **kwargs):
        self.mkdir(data_dir)
        pos_data_dir = os.path.join(data_dir,'positive')
        neg_data_dir = os.path.join(data_dir,'negative')
        pos_image_dir = os.path.join(pos_data_dir,'Images')
        neg_image_dir = os.path.join(neg_data_dir,'Images')
        pos_data_file = os.path.join(pos_data_dir,'dataInfo.txt')
        neg_data_file = os.path.join(neg_data_dir,'dataInfo.txt')

        self.mkdir([pos_data_dir, neg_data_dir, pos_image_dir, neg_image_dir])
        pos_csv_file = open(pos_data_file,'w')
        neg_csv_file = open(neg_data_file,'w')
        pos_csv_writer = csv.writer(pos_csv_file)
        neg_csv_writer = csv.writer(neg_csv_file)
        pos_csv_writer.writerow(['PatchFilePath','Theta','Path', 'GraspHeight', 'GraspWidth'])
        neg_csv_writer.writerow(['PatchFilePath','Theta','Path', 'GraspHeight', 'GraspWidth'])
        for idx,d in enumerate(dataset):
            print('Grasp #{} Grasp Folder: {} Success: {}'.format(idx, d[0], d[1]['grasp_result']))
            grasp_datapoints = self.process_datapoint(d, **kwargs)
            if grasp_datapoints[0] is None:
                continue
            #grasp_datapoints = self.get_labeled_datapoints(grasp_datapoints)
            for gd in grasp_datapoints:
                success = gd['success']
                theta_label = gd['theta']
                image = gd['image']
                im_name = gd['image_name']
                h = gd['h']
                w = gd['w']
                if success:
                    csv_writer = pos_csv_writer
                    image_dir = pos_image_dir
                else:
                    csv_writer = neg_csv_writer
                    image_dir = neg_image_dir
                csv_writer.writerow([im_name, theta_label, '/'.join(d[0].split('/')[3:]), h, w])
                cv2.imwrite(os.path.join(image_dir, im_name), image)

    def split_train_val(self):
        n_dataset = len(self.grasp_dirs)
        n_train = int(n_dataset*self.train_fr)
        n_val = n_dataset-n_train
        random_idxs = np.random.permutation(n_dataset)
        train_idxs = random_idxs[:n_train]
        val_idxs = random_idxs[n_train:]
        self.train_dataset = self.grasp_dirs[train_idxs]
        self.val_dataset = self.grasp_dirs[val_idxs]

    def randomize_dataset(self, dataset):
        n_dataset = len(dataset)
        random_idxs = np.random.permutation(n_dataset)
        return dataset[random_idxs]

    def parse_grasp_data(self, grasp_root_dir, robot_filters=[''], env_filters=[''], grasp_filters=[GRASP_SUBNAME]):
        grasp_dirs = []
        env_stats = []
        robot_ids = os.listdir(grasp_root_dir)
        robot_ids = self.filter_list(robot_ids, robot_filters)
        for robot_id in robot_ids:
            grasp_dir_robot = os.path.join(grasp_root_dir,robot_id)
            grasp_envs = os.listdir(grasp_dir_robot)
            grasp_envs = self.filter_list(grasp_envs, env_filters)
            for grasp_env in grasp_envs:
                grasp_dir_robot_env = os.path.join(grasp_dir_robot,grasp_env)
                print(grasp_dir_robot_env)
                grasp_attempts = os.listdir(grasp_dir_robot_env)
                grasp_attempts = self.filter_list(grasp_attempts, grasp_filters)
                ge_succ = 0
                ge_total = 0
                pos_attempts = []
                neg_attempts = []
                for grasp_attempt in grasp_attempts:
                    grasp_dir_robot_env_attempt = os.path.join(grasp_dir_robot_env,grasp_attempt)
                    grasp_data_file = os.path.join(grasp_dir_robot_env_attempt, DATA_FILE)
                    if not os.path.isfile(grasp_data_file):
                        continue
                    try:
                        grasp_data = pkl.load(open(grasp_data_file, 'rb'))
                    except:
                        print('Error in file: {}'.format(grasp_data_file))
                        continue
                    contains_keys = self.check_keys(ESSENTIAL_KEYS, grasp_data.keys())
                    if not contains_keys:
                        continue
                    isvalid = grasp_data[VALID_KEY]
                    if not isvalid:
                        continue
                    success = grasp_data[SUCCESS_KEY]
                    if success:
                        pos_attempts.append([grasp_dir_robot_env_attempt, grasp_data])
                        ge_succ += 1
                    else:
                        neg_attempts.append([grasp_dir_robot_env_attempt, grasp_data])
                    ge_total += 1
                g_success_rate = float(ge_succ)/(ge_total+0.00001)
                if g_success_rate>=MIN_SUCCESS_RATE:
                    grasp_dirs = grasp_dirs+pos_attempts
                    grasp_dirs = grasp_dirs+neg_attempts
                else:
                    ge_total_old = ge_total
                    ge_fail_pred = float(ge_succ)*(1-MIN_SUCCESS_RATE)/(MIN_SUCCESS_RATE)
                    grasp_dirs = grasp_dirs+pos_attempts
                    grasp_dirs = grasp_dirs+neg_attempts[:int(ge_fail_pred)]
                    ge_fail = len(neg_attempts[:int(ge_fail_pred)])
                    ge_total = ge_succ+ge_fail
                    print('Balancing...: ', grasp_dir_robot_env, ge_succ, ge_total_old-ge_succ, g_success_rate, ge_fail)
                env_stats.append([grasp_dir_robot_env, ge_succ, ge_total, float(ge_succ)/(ge_total+0.001)])
        return grasp_dirs

    def filter_names(self, name, name_filters):
        contains_filter = False
        for f in name_filters:
            if f in name:
                contains_filter = True
                break
        return contains_filter

    def filter_list(self, name_list, name_filters):
        new_list = []
        for l in name_list:
            for f in name_filters:
                if f in l:
                    new_list.append(l)
                    break
        return new_list

    def process_datapoint(self, datapoint, display=False, display_msec=500):
        img_path = os.path.join(datapoint[0], IMAGE_FILE)
        if os.path.isfile(img_path):
            datapoint = copy(datapoint)
            datapoint[1]['image'] = cv2.imread(img_path)
            datapoint[1]['data_path'] = datapoint[0]
            datapoint = datapoint[1]
        else:
            return [None]
        image = datapoint['image']
        w, h = datapoint['grasp_x_img'], datapoint['grasp_y_img']
        t = datapoint['grasp_theta']
        s = datapoint['grasp_result']
        if display==True:
            I = self.draw_rectangle(copy(image), h, w, t, PATCH_SIZE)
            cv2.imshow('image',I)
            cv2.waitKey(display_msec)
        grasp_datapoints = self.data_augment(image, s, h, w, t, PATCH_SIZE, 18)
        return grasp_datapoints

    def data_augment(self, image, suc, h, w, t, ps, na):
        aug_pts = []
        for ang in range(na):
            rot_ang = np.pi*ang/18
            new_t = t+rot_ang
            new_image = self.rotate_image_and_extract_patch(image, -rot_ang, [h,w], ps)
            image_name = self.random_string(10)+'.jpg'
            aug_pts.append({'image':new_image, 'theta':new_t, 'success':suc, 'image_name':image_name, 'h':h, 'w':w})
        return aug_pts

    def random_string(self, N):
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(N))

    def rotate_image_and_extract_patch(self, img, angle, center, size):
            angle = angle*180/np.pi
            padX = [img.shape[1] - center[1], center[1]]
            padY = [img.shape[0] - center[0], center[0]]
            imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
            imgR = scipy.misc.imrotate(imgP, angle)
            #imgR = ndimage.rotate(imgP, angle, reshape=False, order =1)
            half_size = int(size/2)
            return imgR[padY[0] + center[0] - half_size: padY[0] + center[0] + half_size, padX[0] + center[1] - half_size : padX[0] + center[1] + half_size, :]

    def draw_rectangle(self, I, h, w, t, gsize=100):
        I_temp = I
        grasp_l = gsize/2.5
        grasp_w = gsize/5.0
        grasp_angle = t
        points = np.array([[-grasp_l, -grasp_w],
                           [grasp_l, -grasp_w],
                           [grasp_l, grasp_w],
                           [-grasp_l, grasp_w]])
        R = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                      [np.sin(grasp_angle), np.cos(grasp_angle)]])
        rot_points = np.dot(R, points.transpose()).transpose()
        im_points = rot_points + np.array([w,h])
        cv2.line(I_temp, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0,255,0), thickness=5)
        cv2.line(I_temp, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0,0,255), thickness=5)
        cv2.line(I_temp, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0,255,0), thickness=5)
        cv2.line(I_temp, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0,0,255), thickness=5)
        return I_temp

    def check_keys(self, e_keys, g_keys):
        all_present = True
        for key in e_keys:
            if key not in g_keys:
                all_present = False
                break
        return all_present

    def load_dataset(self, grasp_dirs):
        dataset = []
        for gd in grasp_dirs:
            img_path = os.path.join(gd[0], IMAGE_FILE)
            depth_path = os.path.join(gd[0], DEPTH_FILE)
            if os.path.isfile(img_path) and os.path.isfile(depth_path):
                datapoint = copy(gd)
                datapoint[1]['image'] = cv2.imread(img_path)
                datapoint[1]['depth'] = cv2.imread(depth_path)
                datapoint[1]['data_path'] = datapoint[0]
                dataset.append(datapoint[1])
        return dataset

    def mkdir(self, directories):
        if not hasattr(directories, '__iter__'):
            directories = [directories]
        for directory in directories:
            if os.path.exists(directory):
                tmp = tempfile.mktemp(dir=os.path.dirname(directory))
                shutil.move(directory, tmp)
                shutil.rmtree(tmp)
            os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dataset_path', type=str, default=DATA_ROOT_DIR, help='Path to home grasping data')
    parser.add_argument('--patch_dataset_path', type=str, default='.tmp', help='Path to patch dataset')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='ratio of training data')
    parser.add_argument('--display', type=int, default=1, help='1 if you want to display is random sequence; 0 otherwise')
    parser.add_argument('--msec', type=int, default=500, help='milliseconds between display')

    ## Parse arguments
    args = parser.parse_args()
    print('\n\n#######################\n## Loading Home Data ##\n#######################\n')
    D = DataHandler(args.home_dataset_path)
    D.random_val_split(D.train_dataset, args.train_fraction)
    print('\n\n###########################\n## Writing Patch Dataset ##\n###########################\n')
    D.write_train_val(args.patch_dataset_path, display=bool(args.display), display_msec=args.msec)

if __name__=='__main__':
    main()