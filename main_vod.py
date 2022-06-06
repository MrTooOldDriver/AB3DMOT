# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function

import glob

import matplotlib;

from vod import KittiLocations, FrameDataLoader, FrameTransformMatrix, homogeneous_transformation, homogeneous_coordinates

matplotlib.use('Agg')
import os, numpy as np, time, sys, argparse
from AB3DMOT_libs.utils import Config, get_subfolder_seq, initialize
from AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_results, save_affinity
from scripts.post_processing.combine_trk_cat import combine_trk_cat
from xinshuo_io import mkdir_if_missing, save_txt_file
from xinshuo_miscellaneous import get_timestring, print_log


def main_per_cat(cfg, frames, log, ID_start, load_from_label, save_folder_name):
    print('load_from_label: ', load_from_label)
    if load_from_label:
        det_id2str = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist', 4: 'rider', 5: 'bicycle', 6: 'bicycle_rack', 7: 'human_depiction',
                      8: 'moped_scooter', 9: 'motor', 10: 'truck', 11: 'ride_other', 12: 'vehicle_other', 13: 'ride_uncertain', 14: 'DontCare'}
    else:
        det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    det_str2_id = {v: k for k, v in det_id2str.items()}
    # get data-cat-split specific path
    result_sha = '%s_%s_%s' % (cfg.det_name, save_folder_name, cfg.split)
    # det_root = os.path.join('./data', cfg.dataset, 'tracking','training', 'label_2')
    if load_from_label:
        det_root = os.path.join('./data', cfg.dataset, 'tracking', 'training', 'label_2')
    else:
        det_root = os.path.join('./data', cfg.dataset, 'detection')
    # subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(cfg.dataset, cfg.split)
    data_root = os.path.join('./data', cfg.dataset, 'tracking')
    trk_root = os.path.join(data_root, 'training')
    save_dir = os.path.join(cfg.save_root, result_sha + '_H%d' % cfg.num_hypo);
    mkdir_if_missing(save_dir)

    # create eval dir for each hypothesis
    eval_dir_dict = dict()
    for index in range(cfg.num_hypo):
        eval_dir_dict[index] = os.path.join(save_dir, 'data_%d' % index);
        mkdir_if_missing(eval_dir_dict[index])

    # loop every sequence
    seq_count = 0
    total_time, total_frames = 0.0, 0

    # create folders for saving
    eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
        get_saving_dir(eval_dir_dict, frames[0], save_dir, cfg.num_hypo)

    hw = {'image': (1216, 1936), 'lidar': (720, 1920)}

    subfolder = 'training'

    # initialize tracker
    tracker, frame_list = initialize(cfg, trk_root, save_dir, subfolder, frames[0], 'all', ID_start, hw, log, seq_stop=frames[-1])

    for frame in frames:
        # add an additional frame here to deal with the case that the last frame, although no detection
        # but should output an N x 0 affinity for consistency

        # logging
        print_str = 'processing %s %s: %d/%d, %d/%d   \r' % (result_sha, frames[0], seq_count, \
                                                             len(frames), int(frame) - int(frames[0]), len(frames))
        sys.stdout.write(print_str)
        sys.stdout.flush()

        # tracking by detection
        seq_file = os.path.join(det_root, frame + '.txt')
        calib_file = os.path.join(det_root, frame + '.txt')
        kitti_locations = KittiLocations(root_dir="./data/vod/tracking",
                                         output_dir="example_output")

        frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                     frame_number=frame)
        frame_transform = FrameTransformMatrix(frame_data)

        try:
            dets_frame_from_file, flag = load_detection(seq_file, det_str2_id, load_from_label)
        except OSError:
            continue
        if len(dets_frame_from_file) == 0:
            continue

        dets_frame = get_frame_det(dets_frame_from_file, tracker.calib, load_from_label, frame_transform)

        if not load_from_label:
            # if loading from pointrcnn detection result, do transformation
            dets = dets_frame.get('dets')
            xyz_in_lidar = dets[:, 3:6]
            xyz_in_lidar[:, 2] = xyz_in_lidar[:, 2] - (dets[:, 0] / 2)
            xyz_in_cam = homogeneous_transformation(homogeneous_coordinates(xyz_in_lidar), frame_transform.t_camera_lidar)
            dets[:, 3:6] = xyz_in_cam[:, 0:3]
            dets[:, -1] = -dets[:, -1]

            # TODO WHY I NEED +- pi/2
            dets[:, -1] = dets[:, -1] - np.pi / 2
            for i in range(dets.shape[0]):
                if dets[i, -1] > np.pi:
                    dets[i, -1] = dets[i, -1] - np.pi * 2
                elif dets[i, -1] < -np.pi:
                    dets[i, -1] = dets[i, -1] + np.pi * 2
            dets_frame['dets'] = dets
        # test_array = [0,0,0]
        # test_array = np.array([test_array,test_array])
        # print(homogeneous_transformation(homogeneous_coordinates(test_array), frame_transform.t_camera_lidar))

        since = time.time()
        results, affi = tracker.track(dets_frame, int(frame) - int(frames[0]), frames[0])
        total_time += time.time() - since

        # saving affinity matrix, between the past frame and current frame
        # e.g., for 000006.npy, it means affinity between frame 5 and 6
        # note that the saved value in affinity can be different in reality because it is between the
        # original detections and ego-motion compensated predicted tracklets, rather than between the
        # actual two sets of output tracklets
        save_affi_file = os.path.join(affinity_dir, '%05d.npy' % int(int(frame)))
        save_affi_vis = os.path.join(affinity_vis, '%05d.txt' % int(int(frame)))
        if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
            # save affinity as long as there are tracklets in at least one frame
            np.save(save_affi_file, affi)

            # cannot save for visualization unless both two frames have tracklets
            if affi.shape[0] > 0 and affi.shape[1] > 0:
                save_affinity(affi, save_affi_vis)

        # saving trajectories, loop over each hypothesis
        for hypo in range(cfg.num_hypo):
            save_trk_file = os.path.join(save_trk_dir[hypo], '%05d.txt' % (int(frame)))
            save_trk_file = open(save_trk_file, 'w')
            for result_tmp in results[hypo]:  # N x 15
                save_results(result_tmp, save_trk_file, eval_file_dict[hypo], \
                             det_id2str, int(frame) - int(frames[0]), cfg.score_threshold)
            save_trk_file.close()

        total_frames += 1

    print('ID_start: %d' % ID_start)

    if total_time == 0 or total_frames == 0:
        return ID_start

    for index in range(cfg.num_hypo):
        eval_file_dict[index].close()
        ID_start = max(ID_start, tracker.ID_count[index])

    print_log('%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f' % \
              (cfg.dataset, result_sha, total_time, total_frames, total_frames / total_time, \
               tracker.metric, tracker.thres), log=log)

    return ID_start


def main():
    # load config files
    dataset = 'vod'
    config_path = './configs/%s.yml' % dataset
    cfg, settings_show = Config(config_path)
    load_from_label = False

    # overwrite split and detection method
    # if args.split is not '': cfg.split = args.split
    # if args.det_name is not '': cfg.det_name = args.det_name

    # print configs
    time_str = get_timestring()
    log = os.path.join(cfg.save_root, 'log/log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split))
    mkdir_if_missing(log)
    log = open(log, 'w')
    for idx, data in enumerate(settings_show):
        print_log(data, log, display=False)

    # seq loading method
    seqs_list = []
    file_names = []
    # full_seq_data = open('./data/vod/tracking/ImageSets/full.txt', 'r')
    # seqs = []
    # last_seq = 543
    # for line in full_seq_data:
    #     seq_num = int(line)
    #     if seq_num != last_seq + 1:
    #         seqs_list.append(seqs)
    #         seqs = []
    #     last_seq = seq_num
    #     seqs.append(str(seq_num).rjust(5, '0'))
    clips_list = glob.glob('./clips/*.txt')
    for clip in clips_list:
        file_names.append(clip.split('\\')[-1].split('.')[0])
        seq = []
        with open(clip, 'r') as f:
            for line in f:
                seq.append(line.strip())
        seqs_list.append(seq)

    # global ID counter used for all categories, not start from 1 for each category to prevent different
    # categories of objects have the same ID. This allows visualization of all object categories together
    # without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
    ID_start = 1

    # run tracking for each seqs in seqs_list
    for i in range(len(seqs_list)):
        ID_start = main_per_cat(cfg, seqs_list[i], log, ID_start, load_from_label, file_names[i])

    # combine results for every category
    print_log('\ncombining results......', log=log)
    combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d' % cfg.num_hypo, cfg.num_hypo)
    print_log('\nDone!', log=log)
    log.close()


if __name__ == '__main__':
    main()
