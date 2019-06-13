import os
import shutil
from IPython import embed
import tempfile

HOME_DIR = os.environ['HOME']
CUR_DATA_DIR = os.path.join(HOME_DIR,'Dropbox/Apps/LowCostArmDataUploader/grasping_data')
NEW_DATA_DIR = os.path.join(HOME_DIR,'home_dataset/grasping_data')

def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
        print(dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

def mkdir(directories):
    if not hasattr(directories, '__iter__'):
        directories = [directories]
    for directory in directories:
        if os.path.exists(directory):
            tmp = tempfile.mktemp(dir=os.path.dirname(directory))
            shutil.move(directory, tmp)
            shutil.rmtree(tmp)
        os.makedirs(directory)


mkdir(NEW_DATA_DIR)
robot_ids = os.listdir(CUR_DATA_DIR)
for robot_id in robot_ids:
	grasp_dir_robot = os.path.join(CUR_DATA_DIR,robot_id)
	new_grasp_dir_robot = os.path.join(NEW_DATA_DIR,robot_id)
	mkdir(new_grasp_dir_robot)
	grasp_envs = os.listdir(grasp_dir_robot)
	for grasp_env in grasp_envs:
		grasp_dir_robot_env = os.path.join(grasp_dir_robot,grasp_env)
		key_words = grasp_env.split('_')
		if len(key_words)<2:
			continue
		del(key_words[1])
		new_grasp_env = '_'.join(key_words)
		new_grasp_dir_robot_env = os.path.join(new_grasp_dir_robot,new_grasp_env)
		copyDirectory(grasp_dir_robot_env, new_grasp_dir_robot_env)
		# embed()