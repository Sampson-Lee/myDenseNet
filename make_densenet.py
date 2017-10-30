# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:22:03 2017

@author: Sampson
"""


from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# essential implements of densenet, think twice before modify anything
def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    batch_norm = L.BatchNorm(bottom, in_place=False)
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride, 
                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    if dropout>0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv

def dense_block(bottom, num_filter, dropout):
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
def densenet(data_file=None, mode='train_test', batch_size=64, depth=40, first_output=16, growth_rate=12, dropout=0.2):
    nchannels = first_output
    if mode == 'deploy':
        # deploy.prototxt dont need data layer 
        model = L.Convolution(bottom='data', kernel_size=3, stride=1, num_output=nchannels,
                            pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    else:
        data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
                transform_param=dict(mirror=True,crop_size=32,mean_value=[129,124,112],scale=1))
        model = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
                            pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    N = (depth-4)/3
    for i in range(N):
        model = dense_block(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    for i in range(N):
        model = dense_block(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    for i in range(N):
        model = dense_block(model, growth_rate, dropout)
        nchannels += growth_rate

    model = L.BatchNorm(model, in_place=False)
    model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    model = L.InnerProduct(model, num_output=100, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

    if mode == 'deploy':
        prob = L.Softmax(model)
        return to_proto(prob)
    else:
        loss = L.SoftmaxWithLoss(model, label)
        accuracy = L.Accuracy(model, label)
        return to_proto(loss, accuracy)

def make_net(name='Net'):
    with open(train_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write(str(densenet(data_file=train_file, mode='train', batch_size=64)))

    with open(test_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write(str(densenet(data_file=test_file, mode='test', batch_size=64)))

    with open(deploy_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:32\n')
        f.write('input_dim:32\n')
        f.write(str(densenet(mode='deploy')))

def make_solver(batch_size, epoch_mult, train_sam, test_sam):
    epoch = int(train_sam/batch_size)+1
    max_iter = epoch*epoch_mult
    test_iter = int(test_sam/batch_size)+1
    test_interval = epoch

    s = caffe_pb2.SolverParameter()

    s.train_net = train_dir
    s.test_net.append(test_dir)
    s.test_interval = test_interval
    s.test_iter.append(test_iter)

    s.max_iter = max_iter
    s.type = 'Nesterov'
    s.display = int(epoch/5)
    # oscillation if lr is excessive, overfitting if lr is too small 
    s.base_lr =  0.05
    s.momentum = 0.9 
    s.weight_decay = 0.0001

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.stepvalue.append(int(0.9 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    s.snapshot=5000
    s.snapshot_prefix='./dense40/cifar100_dense40_v1'
    print(s)
    with open(solver_dir, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':
    # this version doesn't have bottleneck and compression

    # the path of data
    train_file = '/data/lixinpeng/DataBase/cifar100/cifar100_train_lmdb'
    test_file = '/data/lixinpeng/DataBase/cifar100/cifar100_test_lmdb'
    # the path of prototxt
    train_dir = './train_densenet40_v1.prototxt'
    test_dir = './test_densenet40_v1.prototxt'
    deploy_dir = './deploy_densenet40_v1.prototxt'
    solver_dir = './solver_densenet40_v1.prototxt'

    make_net(name='densenet40')
    make_solver(batch_size=64, epoch_mult=300, train_sam=50000, test_sam=10000)

    print('ok!')
