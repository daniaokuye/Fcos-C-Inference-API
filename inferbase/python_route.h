//
// Created by 李大冲 on 2019-10-10.
//

#ifndef INFER__PYTHON_ROUTE_H
#define INFER__PYTHON_ROUTE_H

#include <Python.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class python_route {
public:
    python_route(float ratio_h, float ratio_w, int H, int W);

    ~python_route();

    void LoadModel(float ratio_h, float ratio_w, int H, int W);

    void RunModel(int, int, void *);

    void PythonPost(void *, void *, void *, int, int, bool toCanvas,
                    cv::Mat ResImg, int media_id, int frame_id, const char *mac);

    void ParseRet(cv::Mat, float, float);

//    void SendDB(cv::Mat, int, int, const char *);

    void PythonInfer(int batch, int row, int col, void *ipt);

private:
    PyObject *pModule, *pFunc, *postFunc, *selfPost;//, *sendFunc
    std::vector <uchar> buffer;
};


#endif //INFER__PYTHON_ROUTE_H
