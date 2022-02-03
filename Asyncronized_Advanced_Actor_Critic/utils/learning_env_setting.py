import os
import shutil
import sys
from termcolor import colored
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def initial_env_setting (exp_name,is_continue = True):
    '''
    :param exp_name: 실험 이름입니당.
    :param is_continue: 이전의 학습을 이어서 할지 설정합니다.
    :return: path_dict를 반환합니다.
    '''
    path_dict = {}
    path_dict['exp_path'] = os.path.join(os.getcwd(),exp_name)
    if not os.path.isdir(path_dict['exp_path']):
        is_continue = False

    if is_continue:
        pass

    else:
        if os.path.isdir(path_dict['exp_path']):
            print(colored("Are you OK when remove all past learning data? [y/n] : ",color='red'),end = '')
            answer = input()
            if answer == 'y' or 'Y':
                shutil.rmtree(path_dict['exp_path'])
            else:
                sys.exit()

        print(colored("Learning is Started with Fresh Environment", color='cyan'))
        os.makedirs(path_dict['exp_path'],exist_ok=True)
    return path_dict,is_continue

def load_from_check_point(path_dict,model,model_name = None,from_check_point = None,is_continue = True):
    '''
    :param path_dict: 유용한 경로들을 담은 dict입니다. initial_env_setting 함수 에서 생성됩니다.
    :param model: 반드시 기본 모델을 입력해줘야 합니다.
    :param model_name: 모델의 이름을 지정해줘야합니다.
    :param is_continue: 이전의 학습을 이어서 할지 설정합니다.
    :return: model,training_data,start_episode 를 반환합니다.
    '''
    if is_continue:
        check_point_list = os.listdir(path_dict['exp_path'])
        if check_point_list == []:
            training_data = {}
            training_data['total_losses'] = []
            training_data['value_losses'] = []
            training_data['policy_losses'] = []
            training_data['scores'] = []
            training_data['average_term'] = -1
            start_episode = 0
        else:
            episode_list = [int(check_point.split('_')[-1].split('.')[0]) for check_point in check_point_list]
            episode_list.sort()
            if from_check_point == None:
                last_episode = episode_list[-1]
            else:
                if int(from_check_point) in episode_list:
                    last_episode = from_check_point
                else:
                    print(f"check_point_{from_check_point} is not in the check_point list")
                    sys.exit()
            model_path = os.path.join(path_dict['exp_path'],'check_point_' + str(last_episode), model_name + '.pth')
            model.load_state_dict(torch.load(model_path))

            training_data_np = np.load(os.path.join(path_dict['exp_path'],'check_point_' + str(last_episode), 'training_data.npz'))
            training_data = {}
            for key,value in training_data_np.items():
                if training_data[key].shape == ():
                    training_data[key] = value
                else:
                    training_data[key] = list(value)
            start_episode = last_episode + 1
    else:
        training_data = {}
        training_data['total_losses'] = []
        training_data['value_losses'] = []
        training_data['policy_losses'] = []
        training_data['scores'] = []
        training_data['average_term'] = -1
        start_episode = 0

    return model, training_data, start_episode

def save_to_check_point(model,training_data,path_dict,model_name,episode):
    '''
    :param model: 저장하고자 하는 모델
    :param training_data: 이전까지 학습 과정이 담겨있음
    :param path_dict: 경로들을 담은 dict
    :param model_name: 모델 명
    :param episode: episode
    :return: 값을 반환하지 않습니다.
    '''
    check_point_path = os.path.join(path_dict['exp_path'],'check_point_' + str(episode))

    # 디렉토리 생성
    if os.path.isdir(check_point_path):
        shutil.rmtree(check_point_path)
    os.makedirs(check_point_path,exist_ok=True)

    # 트레이닝 데이터 저장
    np.savez_compressed(os.path.join(check_point_path,'training_data.npz')
                                    ,average_term = training_data['average_term']
                                    ,total_losses = training_data['total_losses']
                                    ,policy_losses = training_data['policy_losses']
                                    ,value_losses = training_data['value_losses']
                                    ,scores = training_data['scores'])

    # 모델 저장
    torch.save(model.state_dict(),os.path.join(check_point_path,model_name+'.pth'))

    # 그래프 저장
    fig,axes = plt.subplots(2,1,figsize = (15,12))
    epoch_range = np.arange(1,1+len(training_data['total_losses']))
    axes[0].plot(epoch_range,training_data['total_losses'],color = 'blue',linewidth = 2,label = 'Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    axes[0].set_title('Loss & Score Graph')
    axes[0].legend(loc = 'upper right')
    axes[1].plot(epoch_range,training_data['scores'],color = 'blue',linewidth = 2,label = 'Score')
    axes[1].set_ylabel('Score')
    axes[1].grid()
    axes[1].legend(loc = 'lower right')
    axes[1].set_xlabel(f"Episode (x {training_data['average_term']})")
    fig.savefig(check_point_path + '/training_figure.png')
    plt.close()

    fig,axes = plt.subplots(2,1,figsize = (15,12))
    axes[0].plot(epoch_range,training_data['policy_losses'],color = 'blue',linewidth = 2,label = 'Policy Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    axes[0].set_title('Policy Loss & Value Loss Graph')
    axes[0].legend(loc = 'upper right')
    axes[1].plot(epoch_range,training_data['value_losses'],color = 'blue',linewidth = 2,label = 'Value Score')
    axes[1].set_ylabel('Score')
    axes[1].grid()
    axes[1].legend(loc = 'upper right')
    axes[1].set_xlabel(f"Episode (x {training_data['average_term']})")
    fig.savefig(check_point_path + '/training_figure_detail.png')
    plt.close()







