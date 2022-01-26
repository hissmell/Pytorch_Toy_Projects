import torch
import numpy as np
from termcolor import colored
from torch.autograd import Variable
from torch.distributions import Normal,kl_divergence
import copy


def fit(epoch,model,optimizer,data_loader,phase = 'training',device = 'cuda'):
    '''
    :param epoch:
    :param model:
    :param optimizer:
    :param data_loader: torch.utils.data.DataLoader class
    :param phase: 'training' or 'validation' (default = training)
    :param device: 'cuda' or 'cpu' (default = 'cuda'), if gpu isn't available, device is automatically converted to 'cpu'
    :return: epoch_loss,epoch_accuracy
    '''
    if phase == 'training':
        model.train()
    else:
        model.eval()

    if not torch.cuda.is_available():
        device = 'cpu'
    model.to(device)

    print(colored(f"Epoch : {epoch:4d}  (In {phase} session)",'yellow'))
    total_data_num = len(data_loader.dataset)
    running_total_loss = 0.0
    running_reconstruction_loss = 0.0
    running_regularization_loss = 0.0
    latent_space_size = model.latent_space_size
    for batch_idx,(inputs,_) in enumerate(data_loader):

        batch_size = inputs.size()[0]
        labels = copy.deepcopy(inputs)
        if phase == 'training':
            optimizer.zero_grad()

        if phase == 'training':
            inputs.requires_grad = True
            labels.requires_grad = True
        else:
            inputs.requires_grad = False
            labels.requires_grad = False

        inputs = Variable(inputs)
        labels = Variable(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs,means,variances = model(inputs)

        # 여기에 loss 구현
        reconstruction_loss = torch.sum(torch.square(outputs - labels)) / (28*28)

        standard_normal_distributions = Normal(torch.zeros(size=(batch_size,latent_space_size)).to(device)
                                               ,torch.ones(size=(batch_size,latent_space_size)).to(device))
        target_normal_distributions = Normal(means,torch.sqrt(variances))
        regularization_loss = kl_divergence(p = standard_normal_distributions,q = target_normal_distributions).mean()

        total_loss = reconstruction_loss + regularization_loss

        if phase == 'training':
            total_loss.backward()
            optimizer.step()

        running_total_loss += total_loss.detach()
        running_regularization_loss += regularization_loss.detach()
        running_reconstruction_loss += reconstruction_loss.detach()
    torch.cuda.empty_cache()
    epoch_total_loss = running_total_loss / total_data_num
    epoch_reconstruction_loss = running_reconstruction_loss / total_data_num
    epoch_regularization_loss = running_regularization_loss / total_data_num

    print(colored(f"Total Loss : {epoch_total_loss:.6f}\nReconstruction Loss : {epoch_reconstruction_loss:.6f}  Regularization Loss {epoch_regularization_loss:.6f}",'blue'))
    print()
    return epoch_total_loss,epoch_reconstruction_loss,epoch_regularization_loss

def record_training_data(training_data,phase,epoch_total_loss,epoch_reconstruction_loss,epoch_regularization_loss):
    '''
    :param training_data:
    :param phase: 'training' or 'validation'
    :param epoch_loss:
    :param epoch_accuracy:
    :return:
    '''
    epoch_total_loss = epoch_total_loss.to('cpu')
    epoch_reconstruction_loss = epoch_reconstruction_loss.to('cpu')
    epoch_regularization_loss = epoch_regularization_loss.to('cpu')

    if phase == 'training':
        training_data['train_total_losses'].append(epoch_total_loss)
        training_data['train_reconstruction_losses'].append(epoch_reconstruction_loss)
        training_data['train_regularization_losses'].append(epoch_regularization_loss)
    else:
        training_data['valid_total_losses'].append(epoch_total_loss)
        training_data['valid_reconstruction_losses'].append(epoch_reconstruction_loss)
        training_data['valid_regularization_losses'].append(epoch_regularization_loss)

if __name__ == '__main__':
    B = 4; D = 5
    mu1 = torch.Tensor([[2.,2.,3.],[4.,5.,6.]])
    std1 = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])
    p = torch.distributions.Normal(mu1, std1)
    mu2 = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])
    std2 = torch.Tensor([[1., 2., 3.], [4., 5., 7.]])
    q = torch.distributions.Normal(mu2, std2)

    loss = torch.distributions.kl_divergence(p, q)
    print(loss)
    print(loss.dtype)
    print(loss.size())
