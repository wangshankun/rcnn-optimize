MWget is a "MultiLine" wget for all POSTX System!

原始代码：
wget http://jaist.dl.sourceforge.net/project/kmphpfm/mwget/0.1/mwget\_0.1.0.orig.tar.bz2

tar -xjvf mwget\_0.1.0.orig.tar.bz2

cd mwget\_0.1.0.orig

./configure

make

make install

或者不安装，直接执行 ./src/mwget

###增加了warp封装成库集成代码中，把进度条显示，文件大小等显示log全部注释掉了，不报错就执行正常

chmod a+x build.sh

./build.sh

./src/test


![image](https://github.com/wangshankun/rcnn-optimize/blob/master/mwget_lib/readme.jpg)
