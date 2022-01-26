from utils.analysis import load_model_from,show_images_and_labels,show_image_with_model_output,generate_images_with_range
from utils.dataset import load_mnist_dataset
from models import VAE

exp_name = 'Exp_Latent_Size4_01'
model_name = 'VAE_Latent_Size4'
check_point = 199

train_data_loader,valid_data_loader = load_mnist_dataset(batch_size=8)
model = VAE()
model = load_model_from(model_to=model,model_name=model_name,exp_name=exp_name,check_point=check_point)
model.eval()

images,labels = next(iter(valid_data_loader))
show_images_and_labels(images=images,labels=labels,row=2,col=4)
show_image_with_model_output(images=images,model=model,row=2,col=4)
generate_images_with_range(images=images,labels=labels,model=model,axes=(0,1),per=6)
generate_images_with_range(images=images,labels=labels,model=model,axes=(0,2),per=6)
generate_images_with_range(images=images,labels=labels,model=model,axes=(0,3),per=6)
generate_images_with_range(images=images,labels=labels,model=model,axes=(1,2),per=6)
generate_images_with_range(images=images,labels=labels,model=model,axes=(1,3),per=6)
generate_images_with_range(images=images,labels=labels,model=model,axes=(2,3),per=6)
