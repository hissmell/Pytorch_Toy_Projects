from utils.learning_env_setting import initial_env_setting,load_from_check_point,save_to_check_point
from utils.dataset import load_image_dataset
from utils.train import fit,record_training_data
from models import TestModel
from torch.optim import Adam
import torch

# 학습 환경 설정
data_dir_path = 'C:\\Users\\82102\\PycharmProjects\\ReinforcementLearning\\Toy_project\\Cifar_100\\cifar100_data'
exp_name = 'Test_experiment'
model_name = 'Test_Model'
is_continue = True
model = TestModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_dict,is_continue = initial_env_setting(exp_name=exp_name,is_continue=is_continue)
train_data_loader,valid_data_loader = load_image_dataset(data_dir_path=data_dir_path,image_size=32)
model,training_data,start_epoch = load_from_check_point(path_dict=path_dict,
                                                        model=model,
                                                        model_name=model_name,
                                                        is_continue = is_continue)

# 학습
max_epoch = 100
learning_rate = 1e-4
save_term = 5
optimizer = Adam(model.parameters(),lr = learning_rate)
for epoch in range(start_epoch,max_epoch):
    train_epoch_loss,train_epoch_accuracy = fit(epoch,model,optimizer,train_data_loader,phase='training',device=device)
    valid_epoch_loss,valid_epoch_accuracy = fit(epoch,model,optimizer,valid_data_loader,phase='validation',device=device)

    record_training_data(training_data,phase='training',epoch_loss=train_epoch_loss,epoch_accuracy=train_epoch_accuracy)
    record_training_data(training_data,phase='validation',epoch_loss=valid_epoch_loss,epoch_accuracy=valid_epoch_accuracy)

    if epoch % save_term == (save_term-1):
        save_to_check_point(model=model,training_data=training_data,path_dict=path_dict,model_name=model_name,epoch=epoch)



