//
// Created by 李大冲 on 2019-08-25.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "csrc/engine.h"
#include "dewarp/dewarper.h"
#include "python_route.h"

#ifndef RETINANET_INFER_INFER_2_H
#define RETINANET_INFER_INFER_2_H

#endif //RETINANET_INFER_INFER_2_H

class Infer_RT {
public:
    ~Infer_RT();

    Infer_RT(const char *engine_file, const char *input = NULL,
             int meida_id = 0, const char *mac = NULL, const char *yaml = NULL,
             int device = -1, std::string modes = "perimeter");

    void process_();

    template<typename T>
    void setInfo(T *ptr, const char *input, int device, std::string modes);

    void preprocess();

    void postprocess(int, int, int);

    void postprocess_fowShow();

    void getsrc();

    void checkStop();

    static void *process(void *);

    void run();

    deWarp *dewarper;
    cv::Mat src;
//    std::mutex mtx;
private:
    def_retinanet::Engine *engine;

    void *data_d, *scores_d, *boxes_d, *classes_d;

    int channels, num_det, height, width, slots;

    int run_batch, N, N_s, N_b, n_count, n_post;
    int meida_id;
    const char *mac;//, *yaml
    float h_ratio, w_ratio;
    float *scores, *boxes, *classes;
    python_route *pr;
    bool stop;
    std::string yaml, line;
};