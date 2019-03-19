import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='tiny_yolo')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)

    return arg


def str2bool(v):
    return v.lower() in ('true', '1')

def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed



# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=1, help='# of images in each batch of data')
data_arg.add_argument('--max_batches', type=int, default=40200, help='# max_batches')
data_arg.add_argument('--class_num', type=int, default=80, help='Number of classes')
data_arg.add_argument('--steps', type=list, default=[-1,100,20000,30000], help='steps')
data_arg.add_argument('--momentum', type=float, default=0.9, help='momentum')
data_arg.add_argument('--scales', type=list, default=[.1,10,.1,.1], help='scales')
data_arg.add_argument('--decay', type=float, default=0.0005, help='decay')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=30, help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
train_arg.add_argument('--train_patience', type=int, default=50, help='Number of epochs to wait before stopping train')
train_arg.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
train_arg.add_argument(
    "--checkpoint_dir", type=str, default='./ckpt/', help="directory where model checkpoints are saved"
)
train_arg.add_argument('--weightfile', type=str, default='yolov3-tiny.weights', help='path of the weight file')
train_arg.add_argument('--train_txt', type=str, default='data/train.txt', help='path of the train image')
train_arg.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

# testing params
test_arg = add_argument_group('Testing Params')
test_arg.add_argument('--anchors', type=list, default=[10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319], help='the value of anchors')
test_arg.add_argument('--num_anchors', type=int, default=6, help='Number of anchors')
test_arg.add_argument('--test_txt', type=str, default='data/test.txt', help='path of the train image')
test_arg.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
test_arg.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
test_arg.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")


# other params
other_arg = add_argument_group('Other Params')
other_arg.add_argument('--use_gpu', type=str2bool, default=False, help='Whether to run on GPU')
