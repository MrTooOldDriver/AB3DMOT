import glob
import os
import shutil

from AB3DMOT_libs.utils import get_subfolder_seq

# result_dir = './results/vod/'
result_dir = './results/uvod/'
label_output = './label_output/'

file_names = []
# clips_list = glob.glob('./clips/*.txt')
clips_list = glob.glob('./uvod_clips/*.txt')
for clip in clips_list:
    file_names.append(clip.split('\\')[-1].split('.')[0])


for file_name in file_names:
    src_dir = os.path.join(result_dir, 'label_{0}_all_H1_thres'.format(file_name), 'trk_withid_0')
    folder_list = glob.glob(src_dir + '/*')
    to_path = label_output+file_name
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(folder_list[0], to_path, symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False)