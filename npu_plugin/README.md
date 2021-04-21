# 拿君正T31硬件做例子，采用C++ plugin机制实现异构NPU的支持

## 通过Executor接口把模型的事情与业务事情隔离开,应用层可以专注在业务本身逻辑了

## 模型文件夹除了模型本身的权重，还要实现前后处理，确保一份模型对应一份代码，
### 插件机制好处:
###  1.工程开发这样不容易出差错，前后处理种类繁多并且极具有针对性，
###    写到代码里容易配置错误，干脆利用插件机制，把这些细节限定在在模型本身里面；
###  2.一个深度学习零经验的嵌入式开发者，可以快速从npu demo移植成功
###  3. 隔离了模型，也就隔离了算法，这样算法团队与工程团队解耦，扯皮可能性没有了
###  4. 接口本身简单，算法团队也可以容易实现这些，甚至扩展成Python实现都可以（todolist）
###  5. 模型升级非常容易，只需要替换一个so，或者配置文件修改成新版本模型so即可

