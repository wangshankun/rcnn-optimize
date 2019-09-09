#include "rpc/server.h"
#include <vector>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#define  CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)) ; }


using namespace nvinfer1;
using namespace std;
static const int INPUT_H = 1024;
static const int INPUT_W = 1024;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W;

class Logger : public ILogger {
public:
    void log(ILogger::Severity severity, const char* msg) override {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;
    }
    static Logger& Instance() {
        static Logger gInstance;
        return gInstance;
    }
};

static const string gSampleName = "TensorRT.Res18_BiSeNet";
static Logger gLogger;
static IBuilder* builder;
static INetworkDefinition* network;
static IExecutionContext* context;
static IRuntime* runtime;
static ICudaEngine* engine;
static int maxBatchSize = 1;
static int batchSize = 1;
static void* buffers[2];
static cudaStream_t stream;
static float* out_buf;

int main(int argc, char *argv[])
{
    // Creating a server that listens on port 8080
    rpc::server srv(8080);

    srv.bind("init", [](string const& s)
    {
        cudaSetDevice(0);

        builder = createInferBuilder(gLogger);
        assert(builder != nullptr);
        network = builder->createNetwork();
        auto parser = nvonnxparser::createParser(*network, gLogger);
        int verbosity = (int) ILogger::Severity::kERROR;
        if ( !parser->parseFromFile(s.c_str(), verbosity) )
        {
            return -1;
        }

        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 20);
        builder->setFp16Mode(true);

        engine = builder->buildCudaEngine(*network);
        assert(engine);
        // deserialize the engine
        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);

        CHECK(cudaMalloc(&buffers[0], batchSize * 3  * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[1], batchSize * INPUT_H * INPUT_W * sizeof(float)));

        CHECK(cudaStreamCreate(&stream));

        out_buf = new float[1024*1024];

        return 0;
    });

    srv.bind("infer", [](vector<float> input)
    {
        CHECK(cudaMemcpyAsync(buffers[0], &input[0], input.size() * sizeof(float),\
                       cudaMemcpyHostToDevice, stream));
        context->enqueue(batchSize, buffers, stream, nullptr);

        CHECK(cudaMemcpyAsync(out_buf, buffers[1], 1024 * 1024 * sizeof(float),\
                       cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        vector<float> output(out_buf, out_buf + 1024 * 1024);
        return output;
    });

    srv.bind("destory",[](string const& s)
    {
        network->destroy();
        builder->destroy();
        context->destroy();
        runtime->destroy();
        engine->destroy();

        CHECK(cudaFree(buffers[0]));
        CHECK(cudaFree(buffers[1]));
        cudaStreamDestroy(stream);
        
        delete out_buf;

        return 0;
    });

    // Run the server loop.
    srv.run();

    return 0;
}
