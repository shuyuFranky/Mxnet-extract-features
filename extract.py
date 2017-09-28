import mxnet as mx
import numpy as np
import cv2
import os
import time
import argparse

from collections import namedtuple

try:
    xrange
except NameError:
    xrange = range

Batch = namedtuple('Batch', ['data'])

def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
    img = img[np.newaxis, :]  # extend to (example, channel, heigth, width)
    return img

def get_single_prob(sym, arg_params, aux_params, devs=mx.cpu()):
    mod = mx.mod.Module(symbol=sym, label_names=None, context=devs)
    # single image
    mod.bind(for_training = False,
             data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    img = get_image('val_1000/0.jpg')
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    return prob

def get_mmulti_prob(args, sym, arg_params, aux_params, devs=mx.cpu()):
    ## multi images
    batch_size = args.batch_size
    mod = mx.mod.Module(symbol=sym, label_names=None, context=devs)
    mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,224,224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    for i in range(0, 200/batch_size):
        tic = time.time()
        idx = range(i*batch_size, (i+1)*batch_size)
        img = np.concatenate([get_image('val_1000/%d.jpg'%(j)) for j in idx])
        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy()
        yield prob
        print('batch %d, time %f sec'%(i, time.time()-tic))


def extract(args, sym, arg_params, aux_params, devs=mx.cpu()):
    # extract
    layer_name = args.fc_layer
    all_layers = sym.get_internals()
    sym = all_layers[layer_name]
    mod = mx.mod.Module(symbol=sym, label_names=None, context=devs)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params)

    fh = open(args.data_vallist, "r")
    for ln in fh:
        ln = ln.strip()
        img_name = ln.split(" ")[0]
        img = get_image(os.path.join(args.data_path, img_name))
        mod.forward(Batch([mx.nd.array(img)]))
        out = mod.get_outputs()[0].asnumpy()
        for b in xrange(out.shape[0]):
            fp = open('%s/%s'%(args.save_path, img_name), 'a')
            for f in xrange(out.shape[1]):
                fp.write('%f '%out[b,f])
            fp.write("\n")
            fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_prefix', type=str,
                        help='trained model')
    parser.add_argument('--load_epoch', type=int,
                        help='load model epoch')
    parser.add_argument('--gpus', type=str, default='',
                        help="gpu id for extracting, default None of ''")
    parser.add_argument('--fc_layer', type=str, default='fc_output',
                        help='fc layer, in which extract the features')
    parser.add_argument('--data_vallist', type=str,
                        help="vallist for extracting")
    parser.add_argument('--data_path', type=str,
                        help='data path')
    parser.add_argument('--save_path', type=str,
                        help='features save path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_examples', type=int,
                        help="data examples number")

    args = parser.parse_args()

    # devices for extracting
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # load model
    (prefix, epoch) = (args.model_prefix, args.load_epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    extract(args, sym, arg_params, aux_params, devs)
