#include "ops.h"
extern "C" void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8);

static void MNNExp(float* dst, const float* src, size_t dataSize) {
    int countC8        = (int)dataSize / 8;
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        static float parameters[] = {
            (float)log(2.0f), 1.0f / (float)log(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
        MNNExpC8(dst, src, parameters, countC8);
    }
    int remain = countC8 * 8;
    auto param = log(2.0f);
    float xLimit = 87;
    for (int i = remain; i < dataSize; i++) {
        auto x         = -src[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        
        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        
        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dst[i]  = expBasic * expRemain;
    }
}

static std::pair<int, int> multiThreadDivide(int size, int threadNum)
{
    int sizeDivide = size / threadNum;
    sizeDivide = UP_DIV(sizeDivide, 4) * 4;
    int scheduleNumber = 1;
    if (sizeDivide > 0) {
        scheduleNumber = UP_DIV(size, sizeDivide);
    }
    return std::make_pair(sizeDivide, scheduleNumber);
}

void softmax(const float *srcData, float *dstData, int outside, int channel, int threadNum) 
{
    // Max and sub
    MNN_CONCURRENCY_BEGIN(tId, threadNum)
    {
        const float *srcY = srcData + tId * channel;
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += channel * threadNum, dstY += channel * threadNum) {
            float maxValue = srcY[0];
            {
                int c = 1;
                for (; c < channel; ++c) {
                    float value = srcY[c];
                    if (value > maxValue)
                        maxValue = value;
                }
            }

            for (int c = 0; c < channel; ++c) {
                dstY[c] = -srcY[c] + maxValue;
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    //Exp
    auto schedule = multiThreadDivide(channel * outside, threadNum);
    int  sizeDivide = schedule.first;
    int  scheduleNumber = schedule.second;

    MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
        int start = sizeDivide * (int)tId;
        int realSize = sizeDivide;
        if (tId == scheduleNumber -1 ) {
            realSize = channel * outside - start;
        }
        if (realSize > 0) {
            MNNExp(dstData + start, dstData + start, realSize);
        }
    }
    MNN_CONCURRENCY_END();

    // Sum and div
    MNN_CONCURRENCY_BEGIN(tId, threadNum);
    {
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, dstY += channel * threadNum) {
            // sum
            float sumValue = 0;
            for (int c = 0; c < channel; ++c) {
                sumValue += dstY[c];
            }

            // div
            {
                int c = 0;
                for (; c < channel; ++c) {
                    dstY[c] /= sumValue;
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
}


static bool PairCompare(const std::pair<float, float>& lhs,
                        const std::pair<float, float>& rhs) {
  return lhs.second > rhs.second;
}

static void _Argmax(const float* data, int v_len, int n, float* out) 
{
  std::vector<float> v;
  v.assign(data, data + v_len);

  std::vector<std::pair<float, float> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
  {
    pairs.push_back(std::make_pair(float(i), v[i]));
  }

  std::partial_sort(pairs.begin(), pairs.begin() + n, pairs.end(), PairCompare);

  for (size_t k = 0; k < n; k++)
  {
    out[k] = pairs[k].first; 
  }

  for (size_t k = 0; k < n; k++)
  {
    out[n + k] = pairs[k].second; 
  }
}


void argmax(const float *srcData, float *dstData, int outside, int channel, int n, int threadNum)
{
    int k;
    omp_set_num_threads(threadNum);
    #pragma omp parallel for private(k)
    for(k = 0; k < outside; k++)
    {
        const float *srcY = k * channel + srcData;
        float *dstY       = k * 2 * n   + dstData;
        _Argmax(srcY, channel, n, dstY);
    }
}
