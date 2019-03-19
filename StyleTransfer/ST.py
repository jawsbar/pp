import tensorflow as tf
import numpy as np
import util, model, style_transfer
import os

import argparse

def args_parse():
    desc = "Tensorflow implementation of 'Image Style Transfer Using Convolutional Neural Networks"
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='.', help='The directory where the pre-trained model was saved')
    parser.add_argument('--content', type=str, default='images/st.jpg', help='File path of content image (notation in the paper : p)')
    parser.add_argument('--style', type=str, default='images/test.jpg', help='File path of style image (notation in the paper : a)')
    parser.add_argument('--output', type=str, default='images/G.jpg', help='File path of output image')
    parser.add_argument('--mode', type=str, default='test', help='train/test')
    parser.add_argument('--save_dir', type=str, default='trained', help='The directory where the trained model was saved')

    parser.add_argument('--loss_ratio', type=float, default=1e-3, help='Weight of content-loss relative to style-loss')

    parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'], help='VGG19 layers used for content loss')
    parser.add_argument('--style_layers', nargs='+', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for style loss')

    parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0], help='Content loss for each content is multiplied by corresponding weight')
    parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2,.2,.2,.2,.2],
                        help='Style loss for each content is multiplied by corresponding weight')

    parser.add_argument('--initial_type', type=str, default='content', choices=['random','content','style'], help='The initial image for optimization (notation in the paper : x)')
    parser.add_argument('--max_size', type=int, default=512, help='The maximum width or height of input images')
    parser.add_argument('--content_loss_norm_type', type=int, default=3, choices=[1,2,3], help='Different types of normalization for content loss')
    parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations to run')

    args = parser.parse_args()
    return args

'''
def check_args(args):

    try:
        assert len(args.content_layers) == len(args.content_layer_weights)
    except:
        print ('content layer info and weight info must be matched')
        return None

    try:
        assert len(args.style_layers) == len(args.style_layer_weights)
    except:
        print('style layer info and weight info must be matched')
        return None

    try:
        assert args.max_size > 100
    except:
        print('Too small size')
        return None

    model_file_path = args.model_path + '/' + model.MODEL_FILE
    try:
        assert os.path.exists(model_file_path)
    except:
        print ('There is no %s'%model_file_path)
        return None

    try:
        size_in_KB = os.path.getsize(model_file_path)
        assert abs(size_in_KB - 534904783) < 10
    except:
        print('check file size of \'imagenet-vgg-verydeep-19.mat\'')
        print('there are some files with the same name')
        print('pre_trained_model used here can be downloaded from bellow')
        print('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
        return None

    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s'%args.content)
        return None

    try:
        assert os.path.exists(args.style)
    except:
        print('There is no %s' % args.style)
        return None

    return args
'''

def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

def main():
    # parse arguments
    args = args_parse()
    print(123)
    if args is None:
        exit()

    # initiate VGG19 model
    model_file_path = args.model_path + '/' + model.MODEL_FILE
    vgg_net = model.VGG19(model_file_path)

    # load content image and style image
    content_image = util.load_image(args.content, max_size=args.max_size)
    style_image = util.load_image(args.style, shape=(content_image.shape[1],content_image.shape[0]))

    # initial guess for output
    if args.initial_type == 'content':
        init_image = content_image
    elif args.initial_type == 'style':
        init_image = style_image
    elif args.initial_type == 'random':
        init_image = np.random.normal(size=content_image.shape, scale=np.std(content_image))

    # check input images for style-transfer
    # utils.plot_images(content_image,style_image, init_image)

    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers,args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight


    # open session
    sess = tf.Session()

    # build the graph
    st = style_transfer.StyleTransfer(session = sess,
                                      content_layer_ids = CONTENT_LAYERS,
                                      style_layer_ids = STYLE_LAYERS,
                                      init_image = add_one_dim(init_image),
                                      save_dir = args.save_dir,
                                      content_image = add_one_dim(content_image),
                                      style_image = add_one_dim(style_image),
                                      net = vgg_net,
                                      num_iter = args.num_iter,
                                      loss_ratio = args.loss_ratio,
                                      content_loss_norm_type = args.content_loss_norm_type,
                                      )
    # launch the graph in a session
    if args.mode == 'train':
        result_image = st.update()
    elif args.mode =='test':
        result_image = st.result_test()
    else:
        exit()


    # remove batch dimension
    shape = result_image.shape
    result_image = np.reshape(result_image,shape[1:])

    # save result
    util.save_image(result_image,args.output)

if __name__ == '__main__':

    main()