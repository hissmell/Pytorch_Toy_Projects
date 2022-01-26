import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from copy import deepcopy

def load_model_from(model_to,exp_name,model_name,check_point=None):
    model_path = os.path.join(os.getcwd(),exp_name, 'check_point_' + str(check_point), model_name + '.pth')
    model_to.load_state_dict(torch.load(model_path))
    model_to.eval()
    return model_to

def show_images_and_labels(images,labels,row,col):
    images = images.numpy()
    labels = labels.numpy()

    fig = plt.figure()

    data_num = 1
    for r in range(row):
        for c in range(col):
            image_array = np.uint8(images[data_num-1][0]*255)
            label = labels[data_num-1]

            image = Image.fromarray(image_array,'L')

            axis = fig.add_subplot(row,col,data_num)
            axis.imshow(image)
            axis.set_xlabel(f" Label : {label}")
            axis.set_xticks([])
            axis.set_yticks([])
            data_num += 1

    plt.show()

def show_image_with_model_output(images,model,row,col):
    model.eval()
    model.to('cpu')
    outputs,_,_ = model(images,device = 'cpu')

    images = images.numpy()
    outputs = np.clip(outputs.detach().numpy(),0.0,1.0)

    fig = plt.figure()
    data_num = 1
    axis_num = 1
    for r in range(row):
        for c in range(col):
            image_array = np.uint8(images[data_num-1][0]*255)
            output_array = np.uint8(outputs[data_num-1][0]*255)

            image = Image.fromarray(image_array,'L')
            output = Image.fromarray(output_array,'L')

            axis = fig.add_subplot(2*row,col,axis_num)
            axis.imshow(image)
            axis.set_xlabel(f" Data Number : {data_num}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis_num += 1

            axis = fig.add_subplot(2*row,col,axis_num)
            axis.imshow(output)
            axis.set_xlabel(f" Data Number : {data_num}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis_num += 1
            data_num += 1
    plt.show()

def generate_images_with_range(images,labels,model,axes = (0,1),x_range = (-1,1),y_range = (-1,1),per = 6):
    model.eval()
    model.to('cpu')
    means,variances = model.encoder(images)

    axis_0_array = torch.from_numpy(np.linspace(*x_range,per).astype(np.float32))
    axis_1_array = torch.from_numpy(np.linspace(*y_range,per).astype(np.float32))

    z = deepcopy(means.detach().numpy())
    z = torch.from_numpy(z)
    axis_num = 1
    fig = plt.figure()
    plt.title(f"Latent Feature {axes[0]} and {axes[1]} | Label : {labels.detach().numpy()[0]}")
    plt.xticks([])
    plt.yticks([])
    for axis_0_deviate in axis_0_array:
        z[:,axes[0]] += axis_0_deviate
        for axis_1_deviate in axis_1_array:
            z[:,axes[1]] += axis_1_deviate
            outputs = model.decoder(z)

            outputs = np.clip(outputs.detach().numpy(), 0.0, 1.0)
            output_array = np.uint8(outputs[0][0] * 255)
            output_image = Image.fromarray(output_array, 'L')

            axis = fig.add_subplot(per,per,axis_num)
            axis.imshow(output_image)
            axis.set_xlabel(f" X={axis_0_deviate.numpy():.2f},Y={axis_1_deviate.numpy():.2f}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis_num += 1
    plt.show()