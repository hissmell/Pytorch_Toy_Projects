from utils.learning_env_setting import initial_env_setting,load_from_check_point,save_to_check_point
from utils.game import *
from agents import BasicAgent
from models import ActorCriticModel
from torch.optim import Adam
import gym
import torch

# 학습 하이퍼파라미터
max_episode = 3000
learning_rate = 1e-4
save_term = 200

# 학습 환경 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_name = 'Test_experiment'
model_name = 'TD(0)_ActorCritic'
is_continue = True
model = ActorCriticModel(input_feature_size=4
                        ,latent_space_size=64
                        ,action_space_size=2)
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


for episode in range(start_episode,max_episode):
    observation = env.reset()



    if epoch % save_term == (save_term-1):
        save_to_check_point(model=model, training_data=training_data, path_dict=path_dict, model_name=model_name,
                            episode=epoch)



