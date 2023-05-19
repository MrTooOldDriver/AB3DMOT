import glob
import os
import shutil

from AB3DMOT_libs.utils import get_subfolder_seq

tracking_gt_path = './label_2_with_track_ids/label_2/'
output_path = './vod_tracking_gt/'

clips_list = glob.glob('./clips/*.txt')

missing_label_folder_list = []

for clip in clips_list:
    folder_name = clip.split('\\')[-1].split('.')[0]
    # create folder
    if not os.path.exists(output_path+folder_name):
        os.mkdir(output_path+folder_name)
    # read clip
    with open(clip, 'r') as f:
        lines = f.readlines()
    # copy files
    for line in lines:
        line = line.strip()
        try:
            shutil.copy(tracking_gt_path+line+'.txt', output_path+folder_name)
        except FileNotFoundError:
            print('File not found: %s in %s ' % (line, folder_name))
            missing_label_folder_list.append(folder_name)

print('Missing label folder list: ', set(missing_label_folder_list))