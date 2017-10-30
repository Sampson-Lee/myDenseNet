# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:22:03 2017

@author: Sampson
"""

from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# helper function for common structures
def bn_scale_relu_conv(bottom, ks, knum, stride=1, pad=0, dropout=0):
    _xbn   = L.BatchNorm(bottom, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    _xscale = L.Scale(_xbn, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    _xrelu = L.ReLU(_xscale, in_place=True)
    _xconv  = L.Convolution(_xrelu, kernel_size=ks, stride=stride, weight_filler=dict(type='msra'),
                            bias_term=False,bias_filler=dict(type='constant'), num_output=knum, pad=pad)
    if dropout>0:
        _xconv = L.Dropout(_xconv, dropout_ratio=dropout)
    return _xconv

def conv_bn_scale_relu(bottom, ks, knum, stride=1, pad=0):
    _xconv = L.Convolution(bottom, kernel_size=ks, num_output=knum, stride=stride, weight_filler=dict(type='msra'),
                           bias_term=False, bias_filler=dict(type='constant'), pad=pad)
    _xbn   = L.BatchNorm(_xconv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    _xscale = L.Scale(_xbn, in_place=True,  bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    _xrelu = L.ReLU(_xscale, in_place=True)
    return _xrelu

def basic_conv_block(bottom, conv1num, conv3num, postfix, dropout):
    #1x1 conv
    _x1x1conv = bn_scale_relu_conv(bottom, 1, conv1num)
    #3x3 conv
    _x3x3conv = bn_scale_relu_conv(_x1x1conv, 3, conv3num, pad=1, dropout=dropout)
    #concat
    _xConcat  = L.Concat(bottom, _x3x3conv, name='concat_'+postfix, axis=1)
    return _xConcat

def dense_block(bottom, conv1num, conv3num, blockid, layernum, dropout):
    _xinput = bottom
    for i in range(layernum):
        postfix = '{0}_{1}'.format(blockid, i)
        _xConcat = basic_conv_block(_xinput, conv1num, conv3num, postfix, dropout)
        _xinput = _xConcat

    return _xConcat

def transition_layer(bottom, num_channels):
    # Transition layer: 1x1 conv + average pooling
    _x1x1conv = bn_scale_relu_conv(bottom, 1, num_channels)
    _xpool    = L.Pooling(_x1x1conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return _xpool

def classfication_layer(bottom, num_ouput):
    # Classification layer:  global average pool +  InnerProduct + Softmax
    _xpool = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)
    _fc    = L.InnerProduct(_xpool, num_output= num_ouput ,in_place=False,
                            weight_filler=dict(type='xavier'), bias_term=False,
                            bias_filler=dict(type='constant'))
    return _fc

def densenet(net_cfg=None, data_cfg=None, batch_size=None, mode='train', datafile=None):
    if mode == 'deploy':
        data = L.Input(name='data', ntop=1, shape=dict(dim=data_cfg['imgsize']))                           
    else:
        data, label = L.Data(name='data', source=datafile, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                             transform_param=dict(mirror=True,crop_size=32,mean_value=[0,0,0],scale=1))
        
    if data_cfg['Dataset']=='IMAGENET':
        # 7x7 convolution followed by 3x3 max pooling
        conv1 = conv_bn_scale_relu(data, 7, 64, stride=2, pad=3)
        pool1 = L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        _out = pool1
        
    if data_cfg['Dataset']=='CIFAR':
        # 3x3 convolution
        # conv1 = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
        #                     pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        conv1 = conv_bn_scale_relu(data, 3, 24, stride=1, pad=1)
        _out = conv1

    # Build the denseblock
    dense_block_id = 0
    for i, item in enumerate(net_cfg):
        layer_type = item['Type']
        if layer_type == 'DenseBlock':
            _out = dense_block(_out, item['conv1Num'], item['conv3Num'], dense_block_id, item['layerNum'], item['dropout'])
            dense_block_id +=1
        elif layer_type == 'Transition':
            _out = transition_layer(_out, item['conv1Num'])
        elif layer_type == 'Classification':
            _out = classfication_layer(_out, item['OutputNum'])
        else:
            raise ValueError, 'layer_type not supported'+item['Type']

    #Connect full connected network to softmax
    fc = _out
    if mode == 'deploy':
        prob = L.Softmax(fc, name='prob')
        return to_proto(prob)
    else:
        loss = L.SoftmaxWithLoss(fc, label)
        acc = L.Accuracy(fc, label)
        return to_proto(loss, acc)

def make_net(net_cfg, data_cfg, batch_size, name='Net'):
    with open(train_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write(str(densenet(net_cfg, data_cfg, batch_size, mode='train', datafile=train_file)))

    with open(test_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write(str(densenet(net_cfg, data_cfg, batch_size, mode='train', datafile=test_file)))

    with open(deploy_dir, 'w') as f:
        f.write('name:"{}"\n'.format(name))
        f.write(str(densenet(net_cfg, data_cfg, batch_size, mode='deploy')))   

def make_solver(batch_size, train_sam, test_sam, epoch_mult):
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
    s.base_lr =  0.1
    s.momentum = 0.9 
    s.weight_decay = 0.0001

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.stepvalue.append(int(0.9 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    s.snapshot=5000
    s.snapshot_prefix=snapshot_prefix
    print(s)
    with open(solver_dir, 'w') as f:
        f.write(str(s))
        
if __name__ == '__main__':

    # the path of data
    train_file = '/data/lixinpeng/Public/cifar100_self-preprocess_train_lmdb'
    test_file = '/data/lixinpeng/Public/cifar100_self-preprocess_test_lmdb'
    # the path of prototxt
    train_dir = './train_densenet56_v1.prototxt'
    test_dir = './test_densenet56_v1.prototxt'
    deploy_dir = './deploy_densenet56_v1.prototxt'
    solver_dir = './solver_densenet56_v1.prototxt'
    snapshot_prefix = './dense-BC56/cifar100_dense56_v1'

    batch_size = 100
    DATA = {'imgsize':[1,3,32,32] ,'Dataset':'CIFAR', 'train_sam':50000, 'test_sam':10000}

    # configuration
    # in denseblock, conv1Num is bottleneck (about 4*growthrate)and conv3Num is growthrate
    # in transition, conv1Num is compression, θ = 0.5
    CIF100_190_40 = [
        {'Type': 'DenseBlock', 'layerNum':31, 'conv1Num': 128, 'conv3Num': 40, 'dropout': 0.2}, # 24 + 31 * 40 = 1264
        {'Type': 'Transition', 'conv1Num':632 },    # 1264 * 0.5 = 632
        {'Type': 'DenseBlock', 'layerNum':31, 'conv1Num': 128, 'conv3Num': 40, 'dropout': 0.2}, # 632 + 31 * 40 = 1872
        {'Type': 'Transition', 'conv1Num':936},     # 1872 * 0.5 = 936
        {'Type': 'DenseBlock', 'layerNum':31, 'conv1Num': 128, 'conv3Num': 40, 'dropout': 0.2}, # 936 + 31 * 40 = 2176
        {'Type': 'Transition', 'conv1Num':1088},    # 2176 * 0.5 = 1088
        {'Type': 'Classification', 'OutputNum':100},
    ]
    CIF100_100_24 = [
        {'Type': 'DenseBlock', 'layerNum':8, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 24 + 8 * 24 = 216
        {'Type': 'Transition', 'conv1Num':120 },    # 216 * 0.5 = 120
        {'Type': 'DenseBlock', 'layerNum':16, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 120 + 16 * 24 = 504
        {'Type': 'Transition', 'conv1Num':252},     # 504 * 0.5 = 252
        {'Type': 'DenseBlock', 'layerNum':24, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 252 + 24 * 24 = 828
        {'Type': 'Transition', 'conv1Num':414},    # 828 * 0.5 = 414
        {'Type': 'Classification', 'OutputNum':100},
    ]
    CIF100_56_24 = [
        {'Type': 'DenseBlock', 'layerNum':4, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 24 + 4 * 24 = 120
        {'Type': 'Transition', 'conv1Num':60 },    # 120 * 0.5 = 60
        {'Type': 'DenseBlock', 'layerNum':8, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 60 + 8 * 24 = 252
        {'Type': 'Transition', 'conv1Num':126},     # 252 * 0.5 = 126
        {'Type': 'DenseBlock', 'layerNum':16, 'conv1Num': 96, 'conv3Num': 24, 'dropout': 0.2}, # 126 + 16 * 24 = 510
        {'Type': 'Transition', 'conv1Num':255},    # 510 * 0.5 = 255
        {'Type': 'Classification', 'OutputNum':100},
    ]
   
    make_net(CIF100_56_24, DATA, batch_size, name='densenet56')
    make_solver(batch_size, DATA['train_sam'], DATA['test_sam'], epoch_mult=300)

    print('ok!')