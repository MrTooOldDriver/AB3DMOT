vod_dataset_lidar_path = './vod_dataset_lidar'
seqs_list = []
frames = []
last_seq = 543
full_seq_data = open(vod_dataset_lidar_path +'/ImageSets/full.txt', 'r')
scences_count = 0
for line in full_seq_data:
    seq_num = int(line)
    if seq_num != last_seq + 1:
        seqs_list.append(frames)
        frames = []
    last_seq = seq_num
    frames.append(str(seq_num).rjust(5, '0'))
    #do copying



