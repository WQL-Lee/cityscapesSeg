import argparse

from coco2yolo import convert_coco_json
from cityscapes2coco import convert_cityscapes


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert dataset')
    
    # covert cityscapes format to coco format
    parser.add_argument('--datadir', help="data dir for annotations to be converted", default="../dataset/cityscapes", type=str)
    parser.add_argument('--anndir', help="annotion dir for json files", default='../dataset/cityscapes/annotations', type=str)
    parser.add_argument("--savedir",help= "dir for saving processed dataset", default="../processed_dataset",type=str)
    args=parser.parse_args()
    # convert_cityscapes(args.datadir, args.anndir)
    
    # convert coco format to yolo format
    convert_coco_json(args.anndir,  # directory with *.json
                      args.datadir,
                        args.savedir,
                        use_segments=True,
                        cls91to80=False)