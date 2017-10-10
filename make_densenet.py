from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe


def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    if split == 'train':
        # \u8bad\u7ec3\u7684\u65f6\u5019\u6211\u4eec\u5bf9 BN \u7684\u53c2\u6570\u53d6\u6ed1\u52a8\u5e73\u5747\uff0c\u8bbe\u7f6euse_global_stats = False
        batch_norm = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats = False), in_place=True, 
					param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    else:
        # \u6d4b\u8bd5\u7684\u65f6\u5019\u6211\u4eec\u76f4\u63a5\u662f\u6709\u8f93\u5165\u7684\u53c2\u6570\uff0c\u8bbe\u7f6euse_global_stats = True
        batch_norm = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats = True), in_place = True, 
					param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=True, 
				weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), 
					param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)])
    if dropout>0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv

def add_layer(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv, axis=1)
    return concate

def transition(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return pooling

#change the line below to experiment with different setting
#depth -- must be 3n+4
#first_output -- #channels before entering the first dense block, set it to be comparable to growth_rate
#growth_rate -- growth rate
#dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(data_file, mode='train', batch_size=64, depth=40, first_output=16, growth_rate=12, dropout=0.2):
    
    data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
              transform_param=dict(mean_file="/data/lixinpeng/DataBase/cifar100/mean.binaryproto"))

    nchannels = first_output
    model = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
                        pad=1, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    N = (depth-4)/3
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate


    model = L.BatchNorm(model, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    model = L.InnerProduct(model, num_output=100, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)
    return to_proto(loss, accuracy)

def make_net():
    split = 'train'
    with open('train_densenet40.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
	split = 'train'
        print(str(densenet('/data/lixinpeng/DataBase/cifar100/cifar100_train_lmdb', batch_size=20)), file=f)

    with open('test_densenet40.prototxt', 'w') as f:
	split = 'test'
        print(str(densenet('/data/lixinpeng/DataBase/cifar100/cifar100_test_lmdb', batch_size=20)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = './train_densenet40.prototxt'
    s.test_net.append('./test_densenet40.prototxt')
    s.test_interval = 5000
    s.test_iter.append(500)

    s.max_iter = 100000	#40 epochs for cifar100, 90 epochs for imagenet
    s.type = 'Nesterov'
    s.display = 1

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    s.snapshot=5000
    s.snapshot_prefix='./snap/cifar100_dense40'

    solver_path = './solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net()
    make_solver()

