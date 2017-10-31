# myDenseNet
some research and implenments of DenseNet, thanks for distinguished scholars in github！

### 网络结构

非BC版本，参考[densenet40_v1](http://ethereon.github.io/netscope/#/gist/40b61cb367e59a79229c2322ef9ad582)

BC版本，参考[densenet56_v1](http://ethereon.github.io/netscope/#/gist/524b7fb49b53258a56de6c537294edbd)

不同version仅代表预处理不同

### 参数减少，内存占用增大
densenet每次生成features的outputnum不多（论文中称为growrate），参数量的计算公式为inputnum乘以outputnum，因此参数量小；另一方面，densenet输入的features由之前层累加，数量很多，在参数更新中，前向传播和后向传播产生的features都要保存，因此占用内存巨大，不过有省内存的版本

# experiments
论文：72.45%，减去总体均值，除以总体标准差， padding=4 再 crop=32

40_v1：减去总体均值，效果最好，71.4%

40_v2：减去总体均值+除以总体标准差：效果次之，70.3%

40_v3：每张图片减去均值除以标准差：效果较差，68.0%


## reference
[DenseNet_torch](https://github.com/liuzhuang13/DenseNet)：论文使用的版本，此次实验主要参考对象

[DenseNet_caffe](https://github.com/Tongcheng/DN_CaffeScript) && [Efficient Implementation In Caffe](https://github.com/Tongcheng/caffe/)：环境较复杂，未成功安装此版本caffe，官方步骤如下：
```shell
# usage
removeCmakeDep.sh
mkdir build
cd build
cmake ..
make -j8
make runtest -j8
```

[DenseNet_caffe script](https://github.com/yuanyuanli85/DenseNet-Caffe)：生成模型的脚本，此次实验主要参考对象，但没有内存优化