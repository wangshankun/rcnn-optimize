#define PRE_NMS_NUM  6000      //anchor预选框的个数
#define POST_NMS_NUM 300      //选出的roi个数
#define IOU_THRESH   0.7      //nms重合面积筛选门限
#define RPN_MIN_SIZE 4.0      //过滤anchor box的尺寸
#define SCORE_THRESH 0.5      //anchor分数的门限
#define FEAT_STRIDE  16       //anchor的步长,也是网络模型到最后feature map时候的缩小倍数
#define PREC         1000000 //精度在rpn中得分用整数代替,提高排序速度

#define MAX_FEATURE_H      24//feature map 最大可能的尺寸
#define MAX_FEATURE_W      31
#define OUT_CLS_NUM        2 //识别的种类
#define OUT_BOX_NUM        8 //box坐标
#define RPN_CLS_NUM        2 //前景，背景 两类得分
#define RPN_ANCHOR_CHANNEL 9 //anchor的种类
#define POOLING_SIZE       7 //psroi和ave_pool的尺寸
#define AVE_POOLED         1 //ave pool 输出
#define AVE_PAD            0 //ave pool padding的个数
#define MAX_MAP_AREA       (MAX_FEATURE_H * MAX_FEATURE_W)//feature map的面积
#define MAX_ANCHOR_NUM     (MAX_MAP_AREA * RPN_ANCHOR_CHANNEL)//最大的anchor数量

int FEATURE_SIZE(int input)
{ //zfnet网络输入图，得到对应feature map的输出尺寸
  int conv1 = floor((float)(input-1)/2)+1;
  int pool1 = ceil((float)(conv1-1)/2)+1;
  int conv2 = floor((float)(pool1-1)/2)+1;
  int pool2 = ceil((float)(conv2-1)/2)+1;
  return pool2;
}
