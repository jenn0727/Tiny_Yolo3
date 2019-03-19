
from util import *
from tiny_yolo import tiny_yolo
#from SE_yolo import SE_yolo
from config import get_config
from trainer import Trainer
from util import prepare_dirs

def detect(config, weightfile, imgfile):


    m = tiny_yolo(config)

    #m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = config.class_num
    namesfile = 'data/coco.names'

    '''
    num_classes = 80
    if num_classes == 20:
        namesfile = ''
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    '''
    cuda = torch.cuda.is_available() and config.use_gpu

    if cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((416, 416))
    

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.5, cuda)
    print(boxes)
    finish = time.time()

    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'prediction.jpg', class_names)

def main(config):
    prepare_dirs(config)

    trainer = Trainer(config)

    if config.is_train:
        trainer.train()
    else:
        # load a pre-trained model and test
        trainer.test()

if __name__ == '__main__':

    config, unparsed = get_config()
    main(config)

    '''
    weightfile = 'yolov3-tiny.weights'
    imgfile = 'data/13.jpg'
    detect(config, weightfile, imgfile)
    
    
    if len(sys.argv) == 3:

        weightfile = sys.argv[1]
        imgfile = sys.argv[2]
        detect(weightfile, imgfile)

    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
    '''