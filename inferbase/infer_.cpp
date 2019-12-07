//
// Created by 李大冲 on 2019-08-26.
//
#ifdef __JETBRAINS_IDE__
#define DEFINE_string
#define DEFINE_double
//how it works: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/46101718#46101718
#endif


#include "infer_.h"
#include "python_route.h"
#include <gflags/gflags.h>
#include <thread>
#include <exception>

#define sign(x) ( ((x) <0 )? -1 : ((x)> 0) )

DEFINE_string(ipt , "", "input video file name");
DEFINE_string(opt, "", "output video file name");
DEFINE_string(mac, "", "Mac Id");
DEFINE_string(yaml, "", "path to yaml");
DEFINE_double(media_id, 0.0, "media_id");


unsigned long GetTickCount() {
    // https://blog.csdn.net/guang11cheng/article/details/6865992
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

bool loadKeyParams(std::string yaml, int &s) {
//    std::cout << "load----" << std::endl;
    cv::FileStorage fs(yaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        fprintf(stderr, "%s:%d:loadParams falied. 'yaml' does not exist\n", __FILE__, __LINE__);
        return false;
    }
    std::string data, pic;
    fs["stop"] >> data;
    fs["pic"] >> pic;
    fs.release();
    s = 0;
//    std::cout << "read p----" << std::endl;
    if (pic == "y") {
        s = 1;
        fs.open(yaml, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            fprintf(stderr, "%s:%d:loadParams falied. 'yaml' does not exist\n", __FILE__, __LINE__);
            return false;
        }
        fs << "stop" << data;
        fs << "pic" << "n";
        fs.release();
        //return true;
    }
    if (data == "y") return true;
    data[0] -= 32;
    if (data == "Y")return true;
    return false;
}

Infer_RT::~Infer_RT() {
    if (engine)delete (engine);
    delete[] scores;
    delete[] boxes;
    delete[] classes;

}

Infer_RT::Infer_RT(const char *engine_file, const char *input,
                   int meida_id, const char *mac, const char *cyaml,
                   int device, std::string modes) :
        meida_id(meida_id), mac(mac), stop(false), yaml(cyaml) {
    if (cyaml == NULL) {
        std::cerr << "yaml should be set!" << std::endl;
        throw "error";
    }
    line = yaml;
    line = line.replace(line.find("yaml"), 4, "jpg");
    string fcos = "fcos", ef = engine_file;
    if (ef.find(fcos) > ef.length()) cout << "Using retinanet..." << endl;
    else cout << "Using Fcos..." << endl;
    printf("engine_file: %s\ninput: %s\nmac: %s media id:%d\n",
           engine_file, input, mac, meida_id);
    engine = new def_retinanet::Engine(engine_file);
    setInfo(engine, input, device, modes);
    printf("engine Init location:%p...\n", engine);
}

template<typename T>
void Infer_RT::setInfo(T *ptr, const char *input, int device, std::string modes) {
    printf("mac2: %s media id:%d\n", mac, meida_id);
    std::cout << "p: param_" + to_string(meida_id) + ".yaml" << std::endl;
    auto inputSize = ptr->getInputSize();
    num_det = ptr->getMaxDetections();
    auto batch = ptr->getMaxBatchSize();

    channels = 3;
    run_batch = 3;
    slots = 2;
    height = inputSize[0];
    width = inputSize[1];
    bool saveVideo = false;// false;//
    bool savePhoto = true;// false;//
    dewarper = new deWarp(width, height, slots + 1, modes, saveVideo, savePhoto);
    dewarper->readVideo(input, device);
    h_ratio = dewarper->ratio_h;
    w_ratio = dewarper->ratio_w;
    std::cout << "h_ratio:" << h_ratio << " w_ratio: " << w_ratio <<
              "|h:" << height << " w:" << width << std::endl;
    // Create device buffers
    cudaMalloc(&scores_d, batch * num_det * sizeof(float));
    cudaMalloc(&boxes_d, batch * num_det * 4 * sizeof(float));
    cudaMalloc(&classes_d, batch * num_det * sizeof(float));

    int useful_batch = run_batch * slots;
    scores = new float[useful_batch * num_det];
    boxes = new float[useful_batch * num_det * 4];
    classes = new float[useful_batch * num_det];
    N = 1, n_post = n_count = slots - 1;
    N_s = run_batch * num_det;
    N_b = run_batch * num_det * 4;
    //python interpreter
    pr = new python_route(h_ratio, w_ratio, dewarper->row_out, dewarper->col_out);
    testStar = true;
}


void Infer_RT::getsrc() {
    /*用来将展开切分结果输出，用python检查结果，在异步时输出尺寸会受限制（原因暂不明）*/
    int sz = width * height * channels * run_batch;
    float *data = new float[sz];
    cudaMemcpy(data, dewarper->data, sz * sizeof(float), cudaMemcpyDeviceToHost);
    pr->PythonInfer(run_batch, height, width, data);
    delete[](data);
}

void Infer_RT::preprocess() {
    unsigned long beg = GetTickCount();
    dewarper->process();
    unsigned long end = GetTickCount();
    std::cout << "preProcess: " << end - beg << "ms\n";

}

void Infer_RT::process_() {
    n_count = (n_count + 1) % slots;
    unsigned long beg = GetTickCount();
    dewarper->currentStatus();

    std::cout << "Running inference..." << std::endl;
    vector<void *> buffers = {dewarper->data, scores_d, boxes_d, classes_d};
    engine->infer(buffers, run_batch);
    // Get back the bounding boxes
    cudaMemcpy(scores + n_count * N_s, scores_d, sizeof(float) * num_det * run_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes + n_count * N_b, boxes_d, sizeof(float) * num_det * 4 * run_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(classes + n_count * N_s, classes_d, sizeof(float) * num_det * run_batch, cudaMemcpyDeviceToHost);

    std::cout << "kernel infer: " << GetTickCount() - beg << "ms\n";
}


template<typename T>
int *expand_line(T *input_x, T *input_y, int n_points, int &idx, const int total = 100) {
    // "perimeter top angle",90.0; perimeter_bottom_angle, 30.0;
    int one_third_R = 480, step = 60, *points = new int[total * 2];
    idx = 0;
    for (int i = 0; i < n_points; i++) {
        T delta_w = input_x[(i + 1) % n_points] - input_x[i];
        T delta_h = input_y[(i + 1) % n_points] - input_y[i];
        //all肯定不为0
        float all = sqrt((1.0 * pow(delta_h, 2) + 1.0 * pow(delta_w, 2)));
        //越接近中间线，步长越大
        float local_s = 1.0 * step * 3 * one_third_R / (abs(2 * one_third_R - input_y[i]) + one_third_R);
        int N_step = ceil(all / local_s);
        float sin_ = 1.0 * delta_h / all, cos_ = 1.0 * delta_w / all, local_step = 1.0 * all / N_step;
        //闭合图形，不保留终点，只保留起始点
        for (int j = 0; j < N_step; ++j) {
            int x_ = int((j * local_step) * cos_ + input_x[i]);
            int y_ = int((j * local_step) * sin_ + input_y[i]);
            points[idx] = x_;
            points[total + idx] = y_;
            ++idx;
            if (cos_ >= -1e-3 and cos_ <= 1e-3) break;
        }
    }
    return points;
}

void Infer_RT::cal_src_ploygon(cv::Mat src, cv::Mat dst) {
    /*点映射回原图点方法*/
    int input_X[] = {100, 700, 700, 100}, input_Y[] = {200, 200, 900, 900}, nn_points = 4;
    int n_points = 0, total_N = 100;
    int *ret = expand_line(input_X, input_Y, nn_points, n_points, total_N);

    int output_x[n_points], output_y[n_points];
    int *input_x = ret, *input_y = ret + total_N;
    dewarper->mappingPolygon(n_points, input_x, input_y, output_x, output_y);

    for (int i = 0; i < n_points; i++) {
        cv::line(src, cv::Point(output_x[(i + 1) % n_points], output_y[(i + 1) % n_points]),
                 cv::Point(output_x[i], output_y[i]), cv::Scalar(50, 50, 50), 3);
        cv::line(dst, cv::Point(input_x[(i + 1) % n_points], input_y[(i + 1) % n_points]),
                 cv::Point(input_x[i], input_y[i]), cv::Scalar(50, 50, 50), 3);
    }
    delete[] ret;
    cv::imwrite("src.jpg", src);
    cv::imwrite("dst.jpg", dst);
}

void Infer_RT::postprocess(int No, int eclipse, int basemap) {
    unsigned long beg = GetTickCount();
    n_post = (n_post + 1) % slots;
    dewarper->currentImg();
    std::cout << "postprocess..." << std::endl;
    if (basemap != 0)
        dewarper->saveImg(line, true);
//    if (testStar) {
//        cal_src_ploygon(dewarper->src, dewarper->dst);
//        testStar = false;
//    }
    int base = n_post * N_s;

    pr->PythonPost(dewarper->dst, &boxes[base * 4], &scores[base], &classes[base], run_batch, num_det);
    cv::putText(dewarper->dst, "Frame No: " + std::to_string(No) + ". Eclipse: " + std::to_string(eclipse) + "ms.",
                cv::Point(int(1.2 * width), int(0.2 * height)),
                cv::FONT_HERSHEY_PLAIN, 5.0, cv::Scalar(255, 245, 250), 3);
    //0。1*H ;3.0->5.0;2->3
//    dewarper->saveImg(line);// 2
    pr->SendDB(dewarper->dst, meida_id, dewarper->current_frame, mac);// 1

    unsigned long end = GetTickCount();
    std::cout << "postProcess: " << end - beg << "ms\n";
}

void Infer_RT::postprocess_fowShow() {
    unsigned long beg = GetTickCount();
    n_post = (n_post + 1) % slots;
    dewarper->currentImg();
    int base = n_post * N_s, i = 0, h = 0, w = 0;
    int row = dewarper->dst.rows, col = dewarper->dst.cols;
    int half_h = row / 2, qurty_w = col / 3;

    for (int j = 0; j < run_batch * num_det; j++) {
        if (j / num_det == 1)continue;
        if (j / num_det == 2) {
            h = half_h;
            w = qurty_w;
        } else {
            h = w = 0;
        }
        i = j + base;
        // Show results over confidence threshold
        if (scores[i] >= 0.3) {//and classes[i] == 0
            float x1 = boxes[i * 4 + 0] * w_ratio + w;
            float y1 = boxes[i * 4 + 1] * h_ratio + h;
            float x2 = boxes[i * 4 + 2] * w_ratio + w;
            float y2 = boxes[i * 4 + 3] * h_ratio + h;
            cout << "{" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "} ";
            //scores[i] = classes[i] = 0;
            //for (int k = 0; k < 4; k++)boxes[i * 4 + k] = 0;
            //cout << "Found box {";
            //for (int k = 0; k < 4; k++) cout << boxes[i * 4 + k] << ", ";
            //cout << "}\n";
            // Draw bounding box on image
            cv::rectangle(dewarper->dst, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
        }
    }
    string res = "detections" + to_string(N++) + ".jpg";
    std::cout << "save img to:" << res << std::endl;
    // Write image
    dewarper->saveImg(res);
    unsigned long end = GetTickCount();
    std::cout << "\n postProcessForshow: " << end - beg << "ms\n";
}

//about thread: 1. https://www.cnblogs.com/wangguchangqing/p/6134635.html
//              2. https://blog.csdn.net/hai008007/article/details/80246437
void Infer_RT::run() {
    auto fishes = [&]() {
        process_();
    };
    unsigned long beg = GetTickCount(), t1;
    preprocess();
    std::thread dataPreparation, dataPostparation, isStop;
    int i = 0, saveImg;
    while (true) {
        t1 = GetTickCount();
        isStop = std::thread([=, &saveImg] { stop = loadKeyParams(yaml, saveImg); });
        dataPreparation = std::thread(fishes);//model
        preprocess();//read img
        if (i > 0)dataPostparation.join();//python and show img
        dataPreparation.join();
        dataPostparation = std::thread([=] { postprocess(i, int(GetTickCount() - t1), saveImg); });
        isStop.join();

        std::cout << "--- No." << ++i << " & " << GetTickCount() - t1
                  << " ms ---" << saveImg << std::endl << std::endl;
        if (!dewarper->has_frame or stop)break;
    }
    dataPostparation.join();
    unsigned long end = GetTickCount();
    std::cout << "Mean time: " << 1.0 * (end - beg) / N << "ms " << std::endl;
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_ipt.empty()) {
        std::cerr << "Invalid input video file name" << std::endl;
        return -1;
    }
    if (FLAGS_opt.empty()) {
        std::cerr << "Invalid output video file name" << std::endl;
        return -1;
    }

//    Infer_RT inferss = Infer_RT("/home/user/weight/int8_640_1280.plan", FLAGS_ipt.c_str());
    Infer_RT inferss = Infer_RT("/home/user/project/run_retina/weights/fcos_int8_640_1280.plan",
                                FLAGS_ipt.c_str(), int(FLAGS_media_id), FLAGS_mac.c_str(), FLAGS_yaml.c_str());
    auto mode_order = [&]() {
        int N = 5;
        unsigned long beg = GetTickCount();
        for (int i = 0; i < N; i++) {
            std::cout << std::endl << "----------------------------" << std::endl;
            inferss.preprocess();
            inferss.process_();
            inferss.postprocess(i, 0, 0);
        }
        unsigned long end = GetTickCount();
        std::cout << "All time: " << 1.0 * (end - beg) / N << "ms " << std::endl;
    };
    inferss.run();
//    mode_order();

}

// export DISPLAY=:0.0
// ./infer_ --ipt s.mp4 --opt 'a' --media_id 1 --mac '00-02-D1' --yaml param_1.yaml
