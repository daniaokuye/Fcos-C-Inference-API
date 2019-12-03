//
// Created by 李大冲 on 2019-10-10.
//

//https://blog.csdn.net/ziweipolaris/article/details/83689597
// g++ -o cc cc.cpp  `pkg-config --cflags --libs opencv` -I/usr/include/python3.5 -lpython3.5m
//https://stackoverflow.com/questions/9826311/trying-to-understand-linking-procedure-for-writing-python-c-hybrid
//opencv_demo.cpp
// how to use python class & function in c++: https://blog.csdn.net/sihai12345/article/details/82745350
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //需要放在numpy/arrayobject.h之前
#define Mmax(a, b)a>b?a:b

#include "python_route.h"
//#include <opencv/cv.hpp>
#include <Python.h>
#include <iostream>
#include <vector>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include "base64.h"
size_t init() {
    import_array();
}

python_route::~python_route() {
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(postFunc);
    Py_Finalize();
}

python_route::python_route(float ratio_h, float ratio_w, int H, int W) {
    //int argc, char *argv[]
    //wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    //if (program == NULL) {
    //    fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    //    exit(1);
    //}
    //Py_SetProgramName(program);  /* 不见得是必须的 */
    /* 非常重要，折腾的时间主要是因为这儿引起的【1】 */
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"/home/user/project/run_retina/py_extension\")");
//    PyRun_SimpleString("sys.path.append(\"/srv/fisheye_prj/AI_Server/utils/py_extension\")");

    init();
    LoadModel(ratio_h, ratio_w, H, W);
}

void python_route::LoadModel(float ratio_h, float ratio_w, int H, int W) {
    /* 导入模块和函数，貌似两种方式都可以，不需要加.py，后面回再提到 */
    // PyObject *pName = PyUnicode_DecodeFSDefault("simple_module");
    PyObject *pName = PyUnicode_FromString("simple_module");
    /*这些检查也非常有帮助*/
    if (pName == NULL) {
        PyErr_Print();
        throw std::invalid_argument("Error: PyUnicode_FromString");
    }
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }
    pFunc = PyObject_GetAttrString(pModule, "simple_func");
    if (pFunc == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to PyObject_GetAttrString");
    }

    postFunc = PyObject_GetAttrString(pModule, "box_info");
    sendFunc = PyObject_GetAttrString(pModule, "sendToDatabase");
    PyObject *setFunc = PyObject_GetAttrString(pModule, "set_param");
    if (postFunc == NULL or setFunc == NULL or sendFunc == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to build python instance");
    }
    PyObject *pArgs = PyTuple_New(4);
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", ratio_h));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", ratio_w));
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", H));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", W));
    PyObject *pRetValue = PyObject_CallObject(setFunc, pArgs);

    // Py_DECREF
    Py_DECREF(pName);
    Py_DECREF(setFunc);
    Py_DECREF(pArgs);
    Py_DECREF(pRetValue);
}

void python_route::PythonPost(cv::Mat dst, void *boxes, void *scores, void *classes, int run_batch, int num_det) {
    PyObject *pArgs = PyTuple_New(3);
    npy_intp dims_s[] = {run_batch, num_det};
    PyObject *sValue = PyArray_SimpleNewFromData(2, dims_s, NPY_FLOAT32, scores);
    npy_intp dims_c[] = {run_batch, num_det};
    PyObject *cValue = PyArray_SimpleNewFromData(2, dims_c, NPY_FLOAT32, classes);
    npy_intp dims_b[] = {run_batch, num_det, 4};
    PyObject *bValue = PyArray_SimpleNewFromData(3, dims_b, NPY_FLOAT32, boxes);

    PyTuple_SetItem(pArgs, 0, sValue);
    PyTuple_SetItem(pArgs, 1, cValue);
    PyTuple_SetItem(pArgs, 2, bValue);
    PyObject *pRetValue = PyObject_CallObject(postFunc, pArgs);
    if (pRetValue == NULL) {
        PyErr_Print();
        throw std::invalid_argument("CalllObject return NULL");
    }
    ParseRet(dst, pRetValue);
    // Py_DECREF
    Py_DECREF(pArgs);
    Py_DECREF(pRetValue);
}

void python_route::ParseRet(cv::Mat dst, PyObject *pRetValue) {
    int half_h = dst.rows / 2;

    /* 解析返回结果 */
    PyArrayObject *r1, *r2, *r3, *r4, *r5, *r6, *r7;
    if (!PyArg_UnpackTuple(pRetValue, "ref", 7, 7, &r1, &r2, &r3, &r4, &r5, &r6, &r7)) {
        PyErr_Print();
        throw std::invalid_argument("PyArg_ParseTuple Crash!");
    }
    npy_intp *shape1 = PyArray_SHAPE(r1);
    npy_intp *shape2 = PyArray_SHAPE(r2);
    npy_intp *shape3 = PyArray_SHAPE(r3);
    npy_intp *shape4 = PyArray_SHAPE(r4);
    npy_intp *shape5 = PyArray_SHAPE(r5);
    npy_intp *shape6 = PyArray_SHAPE(r6);
    npy_intp *shape7 = PyArray_SHAPE(r7);
    float *base_r1 = (float *) PyArray_DATA(r1);
    float *base_r2 = (float *) PyArray_DATA(r2);
    float *base_r3 = (float *) PyArray_DATA(r3);
    float *base_r4 = (float *) PyArray_DATA(r4);
    float *base_r5 = (float *) PyArray_DATA(r5);
    float *base_r6 = (float *) PyArray_DATA(r6);
    float *base_r7 = (float *) PyArray_DATA(r7);
    std::vector<float> ids(base_r1, base_r1 + shape1[0]);
    std::vector<float> track(base_r2, base_r2 + shape2[0] * shape2[1]);
    std::vector<float> track_num(base_r3, base_r3 + shape3[0]);
    std::vector<float> box(base_r4, base_r4 + shape4[0] * shape4[1]);
    std::vector<float> statistic(base_r5, base_r5 + shape5[0]);
    std::vector<float> support(base_r6, base_r6 + shape6[0] * shape6[1]);
    std::vector<float> color(base_r7, base_r7 + shape7[0]);
    int cw = shape1[0] > 0 ? shape7[0] / shape1[0] : 3;
    //track line;
    for (auto i = 0, j = 0; i < shape3[0]; i++) {
        int len = int(track_num[i]), id = int(ids[i]);
        int r = int(color[cw * i]), g = int(color[cw * i + 1]), b = int(color[cw * i + 2]);
        float x1 = round(box[4 * i]), y1 = round(box[4 * i + 1]),
                w = round(box[4 * i + 2]), h = round(box[4 * i + 3]);
        int ratio = int(1.0 * w * h / (50 * 100)), style = 2;
        ratio = ratio < 5 ? ratio : 5;
        style = style > ratio ? style : ratio;
        for (auto k = j; k < j + len; k++) {
//            std::cout << cv::Point(track[2 * k], track[2 * k + 1]) << ';';
            if (k == j) {
                cv::circle(dst, cv::Point(track[2 * k], track[2 * k + 1]), 1, cv::Scalar(r, g, b), style);
                continue;
            }
            float status = (track[2 * k - 1] - half_h) * (track[2 * k + 1] - half_h);
            if (status < 0)continue;
            cv::line(dst, cv::Point(track[2 * k - 2], track[2 * k - 1]),
                     cv::Point(track[2 * k], track[2 * k + 1]), cv::Scalar(r, g, b), style);
        }
        j += len;
        cv::putText(dst, std::to_string(id), cv::Point(int(x1) + 6, int(y1) + 6), cv::FONT_HERSHEY_PLAIN,
                    Mmax(2.0, style * 0.8), cv::Scalar(r, g, b), Mmax(1, int(style
                        *0.8)));
        cv::rectangle(dst, cv::Point(x1, y1), cv::Point(x1 + w, y1 + h), cv::Scalar(r, g, b), style);
//        std::cout << "c:" << r << ", " << g << ", " << b << ";";
    }
//    std::cout << std::endl;

    //std::cout << "shape[1]:" << shape1[0] <<
    //          " shape2:" << shape2[0] << "," << shape2[1] <<
    //          " shape3:" << shape3[0] <<
    //          " shape4:" << shape4[0] << "," << shape4[1] <<
    //          " shape5:" << shape5[0] <<
    //          " shape6:" << shape6[0] << "," << shape6[1] <<
    //          std::endl;
    //std::cout << "ids ";
    //for (auto c: ids)std::cout << c << ',';
    //std::cout << std::endl;
    //std::cout << "track ";
    //for (auto c: track)std::cout << c << ',';
    //std::cout << std::endl;
    //std::cout << "track_num ";
    //for (auto c: track_num)std::cout << c << ',';
    //std::cout << std::endl;
    //std::cout << "box ";
    //for (auto c: box)std::cout << c << ',';
    //std::cout << std::endl;
    //std::cout << "statistic ";
    //for (auto c: statistic)std::cout << c << ',';
    //std::cout << std::endl;
    //std::cout << "support ";
//    std::cout << "shape[7]:" << shape7[0] << ":" << cw << "\ncolor:\n";
//    for (auto c: color)std::cout << c << ',';
//    std::cout << std::endl;

}

void python_route::SendDB(cv::Mat dst, int media_id, int frame_id, const char *mac) {
    PyObject *pArgs = PyTuple_New(4);
    cv::Size ResImgSiz = cv::Size(dst.cols * 0.4, dst.rows * 0.4);
    cv::Mat ResImg = cv::Mat(ResImgSiz, dst.type());
    cv::resize(dst, ResImg, ResImgSiz);
    //std::cout << "row:" << ResImg.rows << " col:" << ResImg.cols << " c:" << ResImg.channels() << std::endl;
    //1. 使用numpy格式传输数据
    //npy_intp dims_s[] = {ResImg.rows, ResImg.cols, ResImg.channels()};
    //PyObject *sValue = PyArray_SimpleNewFromData(3, dims_s, NPY_UINT8, ResImg.data);
    //PyTuple_SetItem(pArgs, 0, sValue);
    //1. 使用numpy格式传输数据
    //-------------------------------------
    //2. 使用base64传输数据
    std::vector <uchar> buffer;
    buffer.resize(static_cast<size_t>(ResImg.rows) * static_cast<size_t>(ResImg.cols));
    cv::imencode(".jpg", ResImg, buffer);
    auto *enc_msg = reinterpret_cast<unsigned char *>(buffer.data());
    std::string encoded = base64_encode(enc_msg, buffer.size());
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", encoded.c_str()));
    //2. 使用base64传输数据
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", media_id));
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", frame_id));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("s", mac));
    PyObject *pRetValue = PyObject_CallObject(sendFunc, pArgs);
    if (pRetValue == NULL) {
        PyErr_Print();
        throw std::invalid_argument("SendDB return NULL");
    }
}

void python_route::PythonInfer(int batch, int row, int col, void *ipt) {
    /* 准备输入参数 */
    PyObject *pArgs = PyTuple_New(2);
    npy_intp dims[] = {batch, row, col, 3};

    PyObject *pValue = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32, ipt);
    PyTuple_SetItem(pArgs, 0, pValue);  /* pValue的引用计数被偷偷减一，无需手动再减 */
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 2));    /* 图像放大2倍 */
    /* 调用函数 */
    PyObject *pRetValue = PyObject_CallObject(pFunc, pArgs);
}

void python_route::RunModel(int row, int col, void *ipt) {
    /* 准备输入参数 */
    PyObject *pArgs = PyTuple_New(2);
    npy_intp dims[] = {row, col, 3};
    std::cout << "python 70" << std::endl;
    PyObject *pValue = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, ipt);
    PyTuple_SetItem(pArgs, 0, pValue);  /* pValue的引用计数被偷偷减一，无需手动再减 */
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 2));    /* 图像放大2倍 */
    std::cout << "python 74" << std::endl;
    /* 调用函数 */
    PyObject *pRetValue = PyObject_CallObject(pFunc, pArgs);
    /* 解析返回结果 */
    PyArrayObject *ret_array;
    std::cout << "python 79" << std::endl;
    PyArray_OutputConverter(pRetValue, &ret_array);
    std::cout << "python 81" << std::endl;
    npy_intp *shape = PyArray_SHAPE(ret_array);
    cv::Mat big_img(shape[0], shape[1], CV_8UC3, PyArray_DATA(ret_array));
    cv::imwrite("aa.jpg", big_img);
}

int pymain(int argc, char *argv[]) {
    /* 读图 */
    cv::Mat sml_img = cv::imread("build/1001.jpg");
    if (!sml_img.isContinuous()) { sml_img = sml_img.clone(); }
    int row = sml_img.rows, col = sml_img.cols;
    python_route pr = python_route(0, 0, 0, 0);
    pr.RunModel(row, col, sml_img.data);
    return 0;
}
