import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

def classify_generation(model, npz_file):
    model.eval()
    with torch.no_grad():
    
        # Load the file
        data = np.load(npz_file)
        # print("data['arr_0'].shape: ", data['arr_0'].shape)

        count_correct = 0
        count_total = 0
        count_target = 0
        for i, (img_np, label) in enumerate(zip(data['arr_0'], data['arr_1'])):
            # Convert numpy array to PIL Image
            img_pil = Image.fromarray(img_np.squeeze().astype(np.uint8))

            # Apply transform
            img = transform(img_pil)

            # Add batch dimension
            img = img.unsqueeze(0).cuda()
            print("img.shape: ", img.shape)

            # Forward pass
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)

            count_total += 1
            if(predicted == label):
                count_correct += 1
            if(predicted == 0):
                count_target += 1

    return count_target, count_correct, count_total

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Modify ResNet18 for MNIST (1 input channel instead of 3, and 10 classes)
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for 1 channel
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes
model = model.cuda()

npz_file = '/vast/mp5847/minimal-diffusion/trained_models_UNet_4_epochs=100_unit_sphere_0_tv=1.5_random/UNet_4_mnist-400-sampling_steps-1000_images-class_condn_True.npz'
#load pretrained model
model.load_state_dict(torch.load('mnist_resnet18_20.pth'))

count_target, count_correct, count_total = classify_generation(model, npz_file)

print(count_target, count_correct, count_total)
print("count_target/count_total: ", count_target/count_total)
print("count_correct/count_total: ", count_correct/count_total)

    