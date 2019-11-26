#include <iostream>
#include "box_tracking.h"

using namespace std;


extern "C"
{

BoxTracker *new_tracker(float iou_cost_weight_, float cost_th_, int max_mismatch_times_) {
    return new BoxTracker(float(iou_cost_weight_), float(cost_th_), max_mismatch_times_);
}

int *tracking_Frame_Hungarian(BoxTracker *new_tracker, int *detection_rects, int box_num, int img_w_, int img_h_,
                              int *cutline, int line_num) {
    vector<int> cut_lines;
    Rect box;
    vector <Rect> detections;
    for (int i = 0; i < box_num; i++) {
        box.x = detection_rects[0 + 4 * i];
        box.y = detection_rects[1 + 4 * i];
        box.width = detection_rects[2 + 4 * i];
        box.height = detection_rects[3 + 4 * i];
        detections.push_back(box);
    }
    for (int i = 0; i < line_num; i++)
        cut_lines.push_back(cut_lines[i]);
    vector<int> result = new_tracker->tracking_Frame_Hungarian(detections, img_w_, img_h_, cut_lines);
    const int num_box = result.size();
    int *tracking_result = new int[num_box + 100];
    for (int i = 0; i < result.size(); i++) {
        tracking_result[i] = result[i];

    }
    for (int i = result.size(); i < result.size() + 100; i++) {
        tracking_result[i] = -1;

    }

    return tracking_result;
}
void cancelledID(int *ids, int id_num) {
    /*最简单的方式：边界处不在保留框：暂时消失的；切线附近的*/

}

}
