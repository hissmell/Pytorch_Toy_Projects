from utils.learning_env_setting import initial_env_setting,load_from_check_point,save_to_check_point
from utils.game import *
from agents import BasicAgent
from models import TestModel
from torch.optim import Adam
import gym
import torch

# 학습 하이퍼파라미터
max_epoch = 100
learning_rate = 1e-4
save_term = 200

# 학습 환경 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_name = 'Test_experiment'
model_name = 'TD(0)_ActorCritic'
is_continue = True
model = TestModel()
optimizer = Adam(model.parameters(),lr = learning_rate)

# 이전 훈련 데이터 로드
path_dict,is_continue = initial_env_setting(exp_name=exp_name,is_continue=is_continue)
model,training_data,start_episode = load_from_check_point(path_dict=path_dict,
                                                        model=model,
                                                        model_name=model_name,
                                                        is_continue = is_continue)


agent = BasicAgent(model,optimizer)
agent.to(device=device)
env = gym.make('CartPole-v1')


for epoch in range(start_epoch,max_epoch):
    train_epoch_loss,train_epoch_accuracy = fit(epoch,model,optimizer,train_data_loader,phase='training',device=device)
    valid_epoch_loss,valid_epoch_accuracy = fit(epoch,model,optimizer,valid_data_loader,phase='validation',device=device)

    record_training_data(training_data,phase='training',epoch_loss=train_epoch_loss,epoch_accuracy=train_epoch_accuracy)
    record_training_data(training_data,phase='validation',epoch_loss=valid_epoch_loss,epoch_accuracy=valid_epoch_accuracy)

    if epoch % save_term == (save_term-1):
        save_to_check_point(model=model, training_data=training_data, path_dict=path_dict, model_name=model_name,
                            episode=epoch)



