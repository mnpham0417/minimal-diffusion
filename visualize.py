import numpy as np
import matplotlib.pyplot as plt

# Replace 'your_file.npz' with the path to your .npz file
npz_file = '/vast/mp5847/minimal-diffusion/trained_models_unit_sphere_generation_gaussian_tv=1.9/UNet_mnist-250-sampling_steps-1000_images-class_condn_True.npz'

# Load the file
data = np.load(npz_file)


for i, (img, label) in enumerate(zip(data['arr_0'], data['arr_1'])):
    plt.imsave(f'./trained_models_unit_sphere_generation_gaussian_tv=1.9/{label}_{i}.png', img.reshape(28, 28), cmap='gray')
    # if(i == 10):
    #     break

