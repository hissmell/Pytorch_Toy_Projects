from utils.learning_env_setting import initial_env_setting,load_from_check_point,save_to_check_point
from utils.dataset import load_mnist_dataset
from utils.train import fit,record_training_data
from models import VAE
from torch.optim import Adam
import torch

# 학습 환경 설정
exp_name = 'Exp_Latent_Size4_01'
model_name = 'VAE_Latent_Size4'
is_continue = True
model = VAE(latent_space_size=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_dict,is_continue = initial_env_setting(exp_name=exp_name,is_continue=is_continue)
train_data_loader,valid_data_loader = load_mnist_dataset()
model,training_data,start_epoch = load_from_check_point(path_dict=path_dict,
                                                        model=model,
                                                        model_name=model_name,
                                                        is_continue = is_continue)

# 학습
max_epoch = 200
learning_rate = 1e-4
save_term = 10
optimizer = Adam(model.parameters(),lr = learning_rate)
for epoch in range(start_epoch,max_epoch):
    train_total_loss,train_reconstruction_loss,train_regularization_loss = fit(epoch,model,optimizer,train_data_loader,phase='training',device=device)
    valid_total_loss,valid_reconstruction_loss,valid_regularization_loss = fit(epoch,model,optimizer,valid_data_loader,phase='validation',device=device)

    record_training_data(training_data
                         ,phase='training'
                         ,epoch_total_loss=train_total_loss
                         ,epoch_reconstruction_loss=train_reconstruction_loss
                         ,epoch_regularization_loss=train_regularization_loss)
    record_training_data(training_data
                         ,phase='validation'
                         ,epoch_total_loss=valid_total_loss
                         ,epoch_reconstruction_loss=valid_reconstruction_loss
                         ,epoch_regularization_loss=valid_regularization_loss)

    if (epoch % save_term) == (save_term-1):
        save_to_check_point(model=model,training_data=training_data,path_dict=path_dict,model_name=model_name,epoch=epoch)



