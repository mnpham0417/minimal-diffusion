import numpy as np
import matplotlib.pyplot as plt

# Replace 'your_file.npz' with the path to your .npz file
npz_file = '/vast/mp5847/minimal-diffusion/trained_models_UNet_inversion_lr=0.1/UNet_mnist-200-sampling_steps-10_images-class_condn_True.npz'

# Load the file
data = np.load(npz_file)


for i, (img, label) in enumerate(zip(data['arr_0'], data['arr_1'])):
    plt.imsave(f'/vast/mp5847/minimal-diffusion/trained_models_UNet_inversion_lr=0.1/{label}_{i}.png', img.reshape(28, 28), cmap='gray')
    # if(i == 10):
    #     break

