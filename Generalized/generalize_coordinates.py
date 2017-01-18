import numpy as np

def get_test_locations(frame_idx, test_boxes):
    i = 0
    is_first_time = True
    result = None
    while (i < test_boxes.shape[0]) and (test_boxes[i][0] <= frame_idx):
        #print i
        if (test_boxes[i][0] == frame_idx):
            if is_first_time:
                result = np.array([test_boxes[i][0:6]])
                #print result
                is_first_time = False
            else:
                result = np.vstack((result, test_boxes[i][0:6]))
                #print result
        i += 1
    return result

overlap_threshold = 0.8
def overlap_area(test_box, restrict_box):

    result = 0
    w = min(test_box[2] + test_box[0], restrict_box[2] + restrict_box[0]) - max(test_box[0], restrict_box[0]) + 1
    h = min(test_box[3] + test_box[1], restrict_box[3] + restrict_box[1]) - max(test_box[1], restrict_box[1]) + 1

    if w>0 and h>0:
        area = (test_box[2] + 1) * (test_box[3] + 1) + \
            (restrict_box[2] + 1) * (restrict_box[3] + 1) - w * h
        result = w * h / area;

    return result

def get_correct_object_idx(pre_frame_location, object_idx, result):
    i = result.shape[0] - 1
    while (i >= 0):
        #if (result[i][0] == pre_frame_location[0]) and (result[i][2] == pre_frame_location[2]) and (result[i][3] == pre_frame_location[3]) \
        #    and (result[i][2] == pre_frame_location[2]) and (result[i][3] == pre_frame_location[3]) :
        if (int(result[i][0]) == int(pre_frame_location[0]) and (overlap_area(pre_frame_location[2:6], result[i][2:6]) > overlap_threshold):
            return result[i][1]
        i -= 1
    return (object_idx + 1)

test_result_boxes = np.loadtxt('KITTI-16small.txt', delimiter=' ')
test_result_boxes_num = test_result_boxes.shape[0]
frame_num = int(test_result_boxes[test_result_boxes_num - 1][0])

for frame_idx in range(2, frame_num+1):
    if ()

	
	
	