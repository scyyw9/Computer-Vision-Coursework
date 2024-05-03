import argparse
from videoProcessor import VideoProcessor

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./videos/test1.mp4', help='source')
    parser.add_argument('--save_path', type=str, default='./outputs', help='outputs directory')
    parser.add_argument('--output', type=str, default='./output1.avi', help='output video')

    # GPU
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--gpu-id", default=0, type=int)

    # YOLOv8 pre-trained weights
    parser.add_argument('--weights', type=str, default='./weights/yolov8n.pt', help='model.pt path')

    # DeepSort pre-trained weights
    parser.add_argument('--tracker_weights', type=str,
                        default='./deep_sort/deep_sort/deep/checkpoint/resnet34_finetuned.pth',
                        help='model.pth path')

    args = parser.parse_args()

    # device
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and args.cuda else "cpu"
    if torch.cuda.is_available() and args.cuda:
        cudnn.benchmark = True

    # Process the input video
    processor = VideoProcessor(args, device)
    processor.process()
