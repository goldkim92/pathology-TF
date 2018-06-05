import argparse
import os
import tensorflow as tf
from model import Network

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',          type=str,   default='train',    help='train or test')
parser.add_argument('--gpu_number',     type=str,   default='0')

parser.add_argument('--data_dir',       type=str,   default=os.path.join('.','data','224'))
parser.add_argument('--log_dir',        type=str,   default='log') # in assets/ directory
parser.add_argument('--ckpt_dir',       type=str,   default='checkpoint') # in assets/ directory
parser.add_argument('--sample_dir',     type=str,   default='sample') # in assets/ directory
parser.add_argument('--test_dir',       type=str,   default='test') # in assets/ directory
parser.add_argument('--assets_dir',     type=str,   required=True) # if assets_dir='aa' -> assets_dir='./assets/aa'

parser.add_argument('--batch_size',     type=int,   default=128)
# parser.add_argument('--image_size',     type=int,   default=28)
parser.add_argument('--input_size',     type=int,   default=112)
parser.add_argument('--image_c',        type=int,   default=3)
parser.add_argument('--label_n',        type=int,   default=3)
parser.add_argument('--nf',             type=int,   default=32) # number of filters

parser.add_argument('--epoch',          type=int,   default=100)
parser.add_argument('--lr',             type=float, default=0.0001) # learning_rate
parser.add_argument('--beta1',          type=float, default=0.9)
#parser.add_argument('--continue_train', type=bool,  default=False)

parser.add_argument('--sample_step',    type=int,   default=300) 
parser.add_argument('--log_step',       type=int,   default=10) 
parser.add_argument('--ckpt_step',      type=int,   default=10) 

args = parser.parse_args()

def main(_):
    # setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    tf.reset_default_graph()
    
    assets_dir = os.path.join('.','assets',args.assets_dir)
    args.log_dir = os.path.join(assets_dir, args.log_dir)
    args.ckpt_dir = os.path.join(assets_dir, args.ckpt_dir)
    args.sample_dir = os.path.join(assets_dir, args.sample_dir)
    args.test_dir = os.path.join(assets_dir, args.test_dir)
    
    # make directory if not exist
    try: os.makedirs(args.log_dir)
    except: pass
    try: os.makedirs(args.ckpt_dir)
    except: pass
    try: os.makedirs(args.sample_dir)
    except: pass
    try: os.makedirs(args.test_dir)
    except: pass

    # run session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Network(sess,args)
        if args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            model.test()
        else: # real_test
            model.real_test()

# run main function
if __name__ == '__main__':
    tf.app.run()