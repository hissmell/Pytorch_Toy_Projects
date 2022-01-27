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
record_per_save_term = 10

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
render = False

average_total_loss = 0.0
average_policy_loss = 0.0
average_value_loss = 0.0
record_term = save_term // record_per_save_term
for episode in range(start_episode,max_episode):
    done = False
    observation = env.reset()

    score = 0.0

    while not done:
        if render:
            env.render()

        action_index = agent.get_action_from_observation(observation=observation)
        next_observation,reward,done,info = env.step(action_index)

        score += reward
        reward = 0.1 if not done or score == 500 else -1

        agent.append_replay_buffer(observation,action_index,reward,next_observation,done)

        observation = next_observation

    running_total_loss,running_policy_loss,running_value_loss = agent.fit()

    training_data['scores'].append(score)
    training_data['total_losses'].append(total_loss)
    training_data['policy_losses'].append(total_policy_loss)
    training_data['value_losses'].append(total_value_loss)




    if epoch % save_term == (save_term-1):
        save_to_check_point(model=model, training_data=training_data, path_dict=path_dict, model_name=model_name,
                            episode=epoch)



