from pathlib import Path

import torch
from torch import nn
import struct
import numpy as np
import scipy as sp

from vocoder.models import base
from config.hparams import wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer

elSize = 4 #change to 2 for fp16

def convert_model(model_fpath: str, default_model_type: str, out_dir: Path):
    # Do all of this on CPU
    device = torch.device("cpu")

    # Load model weights from provided model path
    checkpoint = torch.load(model_fpath, map_location=device)
    model_type = default_model_type
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]

    # Init the model
    try:
        model, _ = base.init_voc_model(model_type, device)
        model = model.eval()
    except NotImplementedError as e:
        print(str(e))
        return

    # Load model state
    model.load_state_dict(checkpoint["model_state"])

    # Define hparams
    if model_type == base.MODEL_TYPE_FATCHORD:
        hparams = wavernn_fatchord
    elif model_type == base.MODEL_TYPE_GENEING:
        hparams = wavernn_geneing
    elif model_type == base.MODEL_TYPE_RUNTIMERACER:
        hparams = wavernn_runtimeracer
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    # Test the Model
    # torch_test_gru(model, checkpoint)
    # torch_test_conv1d(model, checkpoint)
    # torch_test_conv1d_1x(model, checkpoint)
    # torch_test_conv2d(model, checkpoint)
    # torch_test_batchnorm1d(model, checkpoint)

    # Build Model filename and convert
    out_filename = out_dir.joinpath(Path(model_fpath).stem).with_suffix(".bin")
    with open(str(out_filename), 'wb') as f:
        v = struct.pack('@iiii', hparams.res_blocks, len(hparams.upsample_factors), np.prod(hparams.upsample_factors), hparams.pad)
        f.write(v)
        save_resnet(f, model, hparams)
        save_upsample(f, model, hparams)
        save_main(f, model, hparams)

def compress(W, hparams):
    N = W.shape[1]
    W_nz = W.copy()
    W_nz[W_nz!=0]=1
    L = W_nz.reshape([-1, N // hparams.sparse_group, hparams.sparse_group])
    S = L.max(axis=-1)
    #convert to compressed index
    #compressed representation has position in each row. "255" denotes row end.
    (row,col)=np.nonzero(S)
    idx=[]
    for i in range(S.shape[0]+1):
        idx += list(col[row==i])
        idx += [255]
    mask = np.repeat(S, hparams.sparse_group, axis=1)
    idx = np.asarray(idx, dtype='uint8')
    return (W[mask!=0], idx)

def writeCompressed(f, W, hparams):
    weights, idx = compress(W, hparams)
    f.write(struct.pack('@i',weights.size))
    f.write(weights.tobytes(order='C'))
    f.write(struct.pack('@i',idx.size))
    f.write(idx.tobytes(order='C'))
    return


def linear_saver(f, layer, hparams):
    weight = layer.weight.cpu().detach().numpy()

    bias = layer.bias.cpu().detach().numpy()
    nrows, ncols = weight.shape
    v = struct.pack('@iii', elSize, nrows, ncols)
    f.write(v)
    writeCompressed(f, weight, hparams)
    f.write(bias.tobytes(order='C'))

def conv1d_saver(f, layer, hparams):
    weight = layer.weight.cpu().detach().numpy()
    out_channels, in_channels, nkernel = weight.shape
    v = struct.pack('@iiiii', elSize, not(layer.bias is None), in_channels, out_channels, nkernel)
    f.write(v)
    f.write(weight.tobytes(order='C'))
    if not (layer.bias is None ):
        bias = layer.bias.cpu().detach().numpy()
        f.write(bias.tobytes(order='C'))
    return

def conv2d_saver(f, layer, hparams):
    weight = layer.weight.cpu().detach().numpy()
    assert(weight.shape[0]==weight.shape[1]==weight.shape[2]==1) #handles only specific type used in WaveRNN
    weight = weight.squeeze()
    nkernel = weight.shape[0]

    v = struct.pack('@ii', elSize, nkernel)
    f.write(v)
    f.write(weight.tobytes(order='C'))
    return

def batchnorm1d_saver(f, layer, hparams):

    v = struct.pack('@iif', elSize, layer.num_features, layer.eps)
    f.write(v)
    weight=layer.weight.detach().numpy()
    bias=layer.bias.detach().numpy()
    running_mean = layer.running_mean.detach().numpy()
    running_var = layer.running_var.detach().numpy()

    f.write(weight.tobytes(order='C'))
    f.write(bias.tobytes(order='C'))
    f.write(running_mean.tobytes(order='C'))
    f.write(running_var.tobytes(order='C'))

    return

def gru_saver(f, layer, hparams):
    weight_ih_l0 = layer.weight_ih_l0.detach().cpu().numpy()
    weight_hh_l0 = layer.weight_hh_l0.detach().cpu().numpy()
    bias_ih_l0 = layer.bias_ih_l0.detach().cpu().numpy()
    bias_hh_l0 = layer.bias_hh_l0.detach().cpu().numpy()

    W_ir,W_iz,W_in=np.vsplit(weight_ih_l0, 3)
    W_hr,W_hz,W_hn=np.vsplit(weight_hh_l0, 3)

    b_ir,b_iz,b_in=np.split(bias_ih_l0, 3)
    b_hr,b_hz,b_hn=np.split(bias_hh_l0, 3)

    hidden_size, input_size = W_ir.shape
    v = struct.pack('@iii', elSize, hidden_size, input_size)
    f.write(v)
    writeCompressed(f, W_ir, hparams)
    writeCompressed(f, W_iz, hparams)
    writeCompressed(f, W_in, hparams)
    writeCompressed(f, W_hr, hparams)
    writeCompressed(f, W_hz, hparams)
    writeCompressed(f, W_hn, hparams)
    f.write(b_ir.tobytes(order='C'))
    f.write(b_iz.tobytes(order='C'))
    f.write(b_in.tobytes(order='C'))
    f.write(b_hr.tobytes(order='C'))
    f.write(b_hz.tobytes(order='C'))
    f.write(b_hn.tobytes(order='C'))
    return

def stretch2d_saver(f, layer, hparams):
    v = struct.pack('@ii', layer.x_scale, layer.y_scale)
    f.write(v)
    return

savers = { 'Conv1d':conv1d_saver, 'Conv2d':conv2d_saver, 'BatchNorm1d':batchnorm1d_saver, 'Linear':linear_saver, 'GRU':gru_saver, 'Stretch2d':stretch2d_saver }
layer_enum = { 'Conv1d':1, 'Conv2d':2, 'BatchNorm1d':3, 'Linear':4, 'GRU':5, 'Stretch2d':6 }

def save_layer(f, layer, hparams):
    layer_type_name = layer._get_name()
    v = struct.pack('@i64s', layer_enum[layer_type_name], layer.__str__().encode() )
    f.write(v)
    savers[layer_type_name](f, layer, hparams)
    return

def torch_test_gru(model, checkpoint):
    x = 1.+1./np.arange(1,513)
    h = -3. + 2./np.arange(1,513)

    state=checkpoint['model_state']
    rnn1 = model.rnn1

    weight_ih_l0 = rnn1.weight_ih_l0.detach().cpu().numpy()
    weight_hh_l0 = rnn1.weight_hh_l0.detach().cpu().numpy()
    bias_ih_l0 = rnn1.bias_ih_l0.detach().cpu().numpy()
    bias_hh_l0 = rnn1.bias_hh_l0.detach().cpu().numpy()

    W_ir,W_iz,W_in=np.vsplit(weight_ih_l0, 3)
    W_hr,W_hz,W_hn=np.vsplit(weight_hh_l0, 3)

    b_ir,b_iz,b_in=np.split(bias_ih_l0, 3)
    b_hr,b_hz,b_hn=np.split(bias_hh_l0, 3)

    gru_cell = nn.GRUCell(rnn1.input_size, rnn1.hidden_size).cpu()
    gru_cell.weight_hh.data = rnn1.weight_hh_l0.cpu().data
    gru_cell.weight_ih.data = rnn1.weight_ih_l0.cpu().data
    gru_cell.bias_hh.data = rnn1.bias_hh_l0.cpu().data
    gru_cell.bias_ih.data = rnn1.bias_ih_l0.cpu().data

    # hx_ref = hx.clone()
    # x_ref = x.clone()
    #hx_gru = gru_cell(x_ref, hx_ref)

    sigmoid = sp.special.expit
    r = sigmoid( np.matmul(W_ir, x).squeeze() + b_ir + np.matmul(W_hr, h).squeeze() + b_hr)
    z = sigmoid( np.matmul(W_iz, x).squeeze() + b_iz + np.matmul(W_hz, h).squeeze() + b_hz)
    n = np.tanh( np.matmul(W_in, x).squeeze() + b_in + r * (np.matmul(W_hn, h).squeeze() + b_hn))
    hout = (1-z)*n+z*h.squeeze()
    print(hout)

    #hx_gru=hx_gru.detach().numpy().squeeze()
    #dif = hx_gru-hout

    return

def torch_test_conv1d( model, checkpoint ):

    #x=np.matmul((1.+1./np.arange(1,81))[:,np.newaxis], (-3 + 2./np.arange(1,101))[np.newaxis,:])
    x=np.matmul((1.+1./np.arange(1,81))[:,np.newaxis], (-3 + 2./np.arange(1,11))[np.newaxis,:])

    xt=torch.tensor(x[np.newaxis,:,:],dtype=torch.float32)
    weight=model.upsample.resnet.conv_in.weight
    c = torch.nn.functional.conv1d(torch.tensor(xt).type(torch.FloatTensor), weight)
    c1=c.detach().numpy().squeeze()


    w = weight.detach().numpy()
    y = np.zeros([128,6])
    for i1 in range(128):
            for i3 in range(6):
                y[i1,i3] = (x[:,i3:i3+5]*w[i1,:,:]).sum()
    return y

def torch_test_conv1d_1x( model, checkpoint ):

    #x=np.matmul((1.+1./np.arange(1,81))[:,np.newaxis], (-3 + 2./np.arange(1,101))[np.newaxis,:])
    x=np.matmul((1.+1./np.arange(1,129))[:,np.newaxis], (-3 + 2./np.arange(1,11))[np.newaxis,:])

    xt=torch.tensor(x[np.newaxis,:,:],dtype=torch.float32)
    weight = model.upsample.resnet.layers[0].conv1.weight
    c = torch.nn.functional.conv1d(torch.tensor(xt).type(torch.FloatTensor), weight)
    c1=c.detach().numpy().squeeze()

    w = weight.detach().numpy()
    y = np.zeros([128, 10])
    y = np.matmul(w.squeeze(), x)
    return y

def torch_test_conv2d( model, checkpoint ):
    layer = model.upsample.up_layers[1]
    x=np.matmul((1.+1./np.arange(1,81))[:,np.newaxis], (-3 + 2./np.arange(1,21))[np.newaxis,:])

    xt=torch.tensor(x[np.newaxis,:,:],dtype=torch.float32)
    xt = xt.unsqueeze(1)
    c = layer(xt)

    weight = layer.weight
    weight=weight.detach().numpy()
    assert(weight.shape[0]==weight.shape[1]==weight.shape[2]==1)

    xt=xt.squeeze()
    weight = weight.squeeze()
    npad = (weight.size-1)/2

    y = np.zeros(xt.shape)
    for i in range(xt.shape[0]):
        a = np.pad(xt[i,:], (int(npad), int(npad)), mode='constant')
        for j in range(xt.shape[1]):
            y[i,j] = np.sum(a[j:j+9]*weight)

    x = xt.squeeze()
    weight = weight.squeeze()


    return

def torch_test_batchnorm1d( model, checkpoint ):

    layer = model.upsample.resnet.layers[0].batch_norm1
    x=np.matmul((1.+1./np.arange(1,129))[:,np.newaxis], (-3 + 2./np.arange(1,11))[np.newaxis,:])
    xt=torch.tensor(x[np.newaxis,:,:],dtype=torch.float32)

    weight=layer.weight.detach().numpy()
    bias=layer.bias.detach().numpy()
    running_mean = layer.running_mean.detach().numpy()
    running_var = layer.running_var.detach().numpy()

    x1=xt.detach().numpy().squeeze()

    mean = np.mean(x1, axis=0)
    var = np.var(x1, axis=0)
    eps = layer.eps
    y = ((x1[:,0]-running_mean)/(np.sqrt(running_var+eps)))*weight+bias
    #y = ((x1[:,0]-mean[0])/(np.sqrt(var[0]+eps)))*weight+bias

    c = layer(xt)
    return

def save_resnet_block(f, layers, hparams):
    for l in layers:
        save_layer(f, l.conv1, hparams)
        save_layer(f, l.batch_norm1, hparams)
        save_layer(f, l.conv2, hparams)
        save_layer(f, l.batch_norm2, hparams)

def save_resnet( f, model, hparams ):
    try:
        model.upsample.resnet = model.upsample.resnet1  #temp hack
    except: pass

    save_layer(f, model.upsample.resnet.conv_in, hparams)
    save_layer(f, model.upsample.resnet.batch_norm, hparams)
    save_resnet_block( f, model.upsample.resnet.layers, hparams)  #save the resblock stack
    save_layer(f, model.upsample.resnet.conv_out, hparams)
    save_layer(f, model.upsample.resnet_stretch, hparams)
    return

def save_upsample(f, model, hparams):
    for l in model.upsample.up_layers:
        save_layer(f, l, hparams)
    return

def save_main(f, model, hparams):
    if base.get_model_type(model) == base.MODEL_TYPE_FATCHORD:
        save_layer(f, model.I, hparams)
        save_layer(f, model.rnn1, hparams)
        save_layer(f, model.rnn2, hparams)
        save_layer(f, model.fc1, hparams)
        save_layer(f, model.fc2, hparams)
        save_layer(f, model.fc3, hparams)
    elif base.get_model_type(model) == base.MODEL_TYPE_GENEING:
        save_layer(f, model.I, hparams)
        save_layer(f, model.rnn1, hparams)
        save_layer(f, model.fc1, hparams)
        save_layer(f, model.fc3, hparams)
    elif base.get_model_type(model) == base.MODEL_TYPE_RUNTIMERACER:
        save_layer(f, model.I, hparams)
        save_layer(f, model.rnn1, hparams)
        save_layer(f, model.rnn2, hparams)
        save_layer(f, model.rnn3, hparams)
        save_layer(f, model.rnn4, hparams)
        save_layer(f, model.fc1, hparams)
        save_layer(f, model.fc2, hparams)
        save_layer(f, model.fc3, hparams)
        save_layer(f, model.fc4, hparams)
        save_layer(f, model.fc5, hparams)
    else:
        raise NotImplementedError("Provided object is not a valid vocoder model.")
    return