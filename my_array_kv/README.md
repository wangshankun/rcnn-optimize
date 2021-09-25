### 高效落盘的一个kv array库; 本例子是:map<std::array<char, 64>, std::array<float, 64>>

```
使用场景：在通过姓名找到人脸feature(64个float)的数据库中;
人脸底库一般是固定大小十万、五千这种;

原始版本来自stackoverflow一个回答
https://stackoverflow.com/questions/48409391/faster-way-to-read-write-a-stdunordered-map-from-to-a-file

使用到一个字符串转uint64库来自
https://blog.csdn.net/qq_43561345/article/details/116612560

```
