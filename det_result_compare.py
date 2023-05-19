import glob
import os.path

# label_list = glob.glob('./label_output/*/*.txt')
label_list = glob.glob('./label_2_with_track_ids/*/*.txt')
gt_label_path = './data/vod/tracking/training/label_2/'
max_diff = 0
max_diff_ratio = 0
max_diff_seq = ''
total_diff = 0
total_gt_object = 0
total_miss = 0
max_miss = 0
for label in label_list:
    try:
        seq_name = label.split('\\')[-1]
        f = open(label, 'r')
        label_count = len(f.readlines())
        f.close()
        f = open(gt_label_path + seq_name, 'r')
        gt_label_count = len(f.readlines())
        f.close()
        if label_count == gt_label_count:
            continue
        if label_count - gt_label_count < 0:
            total_miss += label_count - gt_label_count
            print('{0} miss {1} objects'.format(seq_name, label_count - gt_label_count))
            max_miss = min(max_miss, label_count - gt_label_count)
            continue
        total_gt_object += gt_label_count
        diff = label_count - gt_label_count
        diff_ratio = diff / gt_label_count
        total_diff += label_count - gt_label_count if label_count > gt_label_count else 0
        print('seq=%s count_diff=%i ratio_diff=%f' % (seq_name, diff, diff_ratio))
        if diff_ratio > max_diff_ratio:
            max_diff_ratio = diff_ratio
            max_diff_seq = label
        max_diff = max(max_diff, diff)
    except FileNotFoundError:
        continue
print('max_diff=%i' % max_diff)
print('max_diff_ratio=%f' % max_diff_ratio)
print('max_diff_seq=%s' % label)
print('total_diff=%i' % total_diff)
print('total_gt_object=%i' % total_gt_object)
print('total_miss=%i' % total_miss)
