# Caffe LSTM拼接成cudnn BLSTM
### cudnn BLSTM的weight排列顺序是：正向的wx，wh然后反向的wx，wh;然后是正向的bx，bh，然后反向bx，bh;
### cudnn BLSTM 四个门顺序是ifco;

### caffe的lstm；weight顺序是wx bx wh；四个门的顺序ifoc
### caffe的reverse lstm时候，除了输入反向以外，来自cudnnblstm的weight都要反向才行；

### caffe的公式I\*Wih+H\*Whh+B而cudnnblstm是I\*Wih+Bih+H\*Whh+Bhh；
### 一个lstm单元比caffe多了一个Bhh；这个在pytorch、TensorFlow都有；
### 也就是说cudnn blstm有八个weight，而caffe需要六个，转成caffe时候
### 把Bih + Bhh的结果放到B里面就好了；

## 上述转换内容在脚本例子中可以看到

* 使用到的reverse 来自CTPN； 包括用两个lstm拼接成一个BLSTM的结构都来自于CTPN
* https://github.com/tianzhi0549/CTPN/blob/master/models/deploy.prototxt
* 另外，transpose在lstm输入数据转换时候会需要，原始caffe没有
