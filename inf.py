import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CityscapesTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(root_dir, "leftImg8bit", "test")) for f in filenames if 'leftImg8bit.png' in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

test_dataset = CityscapesTestDataset('/home/maith/Desktop/cityscapes', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pools = [nn.AdaptiveAvgPool2d(output_size=size) for size in pool_sizes]
        self.conv_blocks = nn.ModuleList([nn.Conv2d(in_channels, 512, 1) for _ in pool_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(512) for _ in pool_sizes])

    def forward(self, x):
        features = [x]
        for pool, conv, bn in zip(self.pools, self.conv_blocks, self.batch_norms):
            pooled = pool(x)
            convolved = conv(pooled)
            upsampled = F.interpolate(convolved, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(bn(upsampled))
        return torch.cat(features, dim=1)

class CascadeFeatureFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, num_classes):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2)
        self.conv_high = nn.Conv2d(high_channels, out_channels, 1)
        self.conv_low_bn = nn.BatchNorm2d(out_channels)
        self.conv_high_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.class_conv = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, low_res_input, high_res_input):
        low_res = self.relu(self.conv_low_bn(self.conv_low(low_res_input)))
        high_res = self.relu(self.conv_high_bn(self.conv_high(high_res_input)))
        
        high_res = F.interpolate(high_res, size=low_res.shape[2:], mode='bilinear', align_corners=False)
        
        result = low_res + high_res
        class_output = self.class_conv(result)
        return result, class_output

class ICNet(nn.Module):
    def __init__(self, num_classes):
        super(ICNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.ppm = PyramidPoolingModule(2048, [1, 2, 3, 6])
        self.cff1 = CascadeFeatureFusion(1024, 4096, 256, num_classes)
        self.cff2 = CascadeFeatureFusion(512, 256, 128, num_classes)

    def forward(self, x):
        layer1 = self.backbone[4](self.backbone[3](self.backbone[2](self.backbone[1](self.backbone[0](x)))))
        layer2 = self.backbone[5](layer1)
        layer3 = self.backbone[6](layer2)
        layer4 = self.backbone[7](layer3)

        ppm_output = self.ppm(layer4)

        cff1_output, class_output1 = self.cff1(layer3, ppm_output)
        cff2_output, class_output2 = self.cff2(layer2, cff1_output)

        final_output = F.interpolate(class_output2, scale_factor=4, mode='bilinear', align_corners=False)  # Scale up to input image size

        return final_output

model = ICNet(num_classes=19)
model.load_state_dict(torch.load('trained_icnet_model_final.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def decode_segmap(image, num_classes=19):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(num_classes):
        idx = image == l
        r[idx] = CITYSCAPES_COLORS[l, 0]
        g[idx] = CITYSCAPES_COLORS[l, 1]
        b[idx] = CITYSCAPES_COLORS[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

denormalize = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def visualize_and_save_predictions(model, device, data_loader, num_images=5, save_dir='/home/maith/Desktop/cityscapes/predictions'):
    model.eval()
    images_so_far = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for images, paths in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = F.interpolate(outputs, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            pred_masks = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if images_so_far >= num_images:
                    return
                
                image = denormalize(images[i].cpu()).numpy().transpose(1, 2, 0)
                pred_mask = pred_masks[i].cpu().numpy()
                decoded_pred_mask = decode_segmap(pred_mask)
                
                pred_mask_image = Image.fromarray(decoded_pred_mask)
                pred_mask_path = os.path.join(save_dir, f'predicted_mask_{os.path.basename(paths[i])}')
                pred_mask_image.save(pred_mask_path)
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Input Image")
                plt.subplot(1, 2, 2)
                plt.imshow(decoded_pred_mask)
                plt.title("Predicted Annotation")
                plt.savefig(os.path.join(save_dir, f'comparison_{os.path.basename(paths[i])}'))
                plt.close()
                
                images_so_far += 1

model.load_state_dict(torch.load('/home/maith/Desktop/cityscapes/trained_icnet_model_final.pth', map_location=device))
visualize_and_save_predictions(model, device, test_loader, num_images=5)