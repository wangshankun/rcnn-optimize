#include <iostream>
#include <vector>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <rpc/client.h>
#include <time.h>

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

using namespace std;

int main(int argc, char** argv) 
{
    // Creating a client that connects to the localhost on port 8080
    rpc::client client("127.0.0.1", 8080);

    cv::Mat image = cv::imread("test.jpg");
    image.convertTo(image, CV_32FC3);
    cv::resize(image, image, cv::Size(1024, 1024));

    image = image / 255.0 * 3.2 - cv::Scalar(1.6f, 1.6f, 1.6f);
    float* dst = new float[1024*1024*3];
    float* src = (float*)(image.data);
    for(int i = 0; i < 1024*1024; i++)
    {
        dst[i] = src[i * 3];
        dst[i + 1024*1024] = src[i * 3 + 1];
        dst[i + 1024*1024*2] = src[i * 3 + 2];
    }

    string model_file = "Tan_Gao_Shui_Split_BiSeNet_1024_SyncBN_FromScratch_2019-06-28_22-42-20_best_bilinear.onnx";
    int ret = client.call("init", model_file).as<int>();


    vector<float> vecImg(dst, dst + 1024*1024*3);
    auto vecRes = client.call("infer", vecImg).as<vector <float>>();
    assert(vecRes.size() == 1024 * 1024);
/*
    float *res_buf = new float[1024 * 1024];
    if (!vecRes.empty())
    {
        memcpy(res_buf, &vecRes[0], vecRes.size()*sizeof(float));
    }
*/
    savefile("res.bin", &vecRes[0], vecRes.size()*sizeof(float));

    return 0;
}
