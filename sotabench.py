import json

import PIL
import torch
import torchvision.transforms as transforms
from torchbench.image_classification import ImageNet

from codebase.networks import NATNet


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """

    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


# Model 1
# Define the transforms need to convert ImageNet data to expected model input
net_config = json.load(open('subnets/imagenet/NAT-M1/net.config'))
if 'img_size' in net_config:
    img_size = net_config['img_size']
else:
    img_size = 224
input_transform = transforms.Compose([
    ECenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
model = NATNet.build_from_config(net_config, pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='NAT-M1',
    paper_arxiv_id='2005.05859',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    model_description="Official weights from the authors of the paper.",
    paper_results={'Top 1 Accuracy': 0.775, 'Top 5 Accuracy': 0.935}
)
torch.cuda.empty_cache()

# Model 2
# Define the transforms need to convert ImageNet data to expected model input
net_config = json.load(open('subnets/imagenet/NAT-M2/net.config'))
if 'img_size' in net_config:
    img_size = net_config['img_size']
else:
    img_size = 224
input_transform = transforms.Compose([
    ECenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
model = NATNet.build_from_config(net_config, pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='NAT-M2',
    paper_arxiv_id='2005.05859',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    model_description="Official weights from the authors of the paper.",
    paper_results={'Top 1 Accuracy': 0.786, 'Top 5 Accuracy': 0.943}
)
torch.cuda.empty_cache()

# Model 3
# Define the transforms need to convert ImageNet data to expected model input
net_config = json.load(open('subnets/imagenet/NAT-M3/net.config'))
if 'img_size' in net_config:
    img_size = net_config['img_size']
else:
    img_size = 224
input_transform = transforms.Compose([
    ECenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
model = NATNet.build_from_config(net_config, pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='NAT-M3',
    paper_arxiv_id='2005.05859',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    model_description="Official weights from the authors of the paper.",
    paper_results={'Top 1 Accuracy': 0.799, 'Top 5 Accuracy': 0.949}
)
torch.cuda.empty_cache()

# Model 4
# Define the transforms need to convert ImageNet data to expected model input
net_config = json.load(open('subnets/imagenet/NAT-M4/net.config'))
if 'img_size' in net_config:
    img_size = net_config['img_size']
else:
    img_size = 224
input_transform = transforms.Compose([
    ECenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
model = NATNet.build_from_config(net_config, pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='NAT-M4',
    paper_arxiv_id='2005.05859',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    model_description="Official weights from the authors of the paper.",
    paper_results={'Top 1 Accuracy': 0.805, 'Top 5 Accuracy': 0.952}
)
torch.cuda.empty_cache()
