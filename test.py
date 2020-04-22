import json
import argparse
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert TF model to PyTorch model and save for easier future loading')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0',
                        help='efficientnet-b{N}, where N is an integer 0 <= N <= 8')
    parser.add_argument('--checkpoint', type=str, default='pretrained_pytorch/efficientnet-b0.pth',
                        help='checkpoint file path')
    args = parser.parse_args()
    print(args)
    model = EfficientNet.from_name(args.model_name)

    state_dict = torch.load(args.checkpoint)
    ret = model.load_state_dict(state_dict, strict=False)

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('examples/simple/img.jpg')).unsqueeze(0)
    print(img.shape) # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('examples/simple/labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
