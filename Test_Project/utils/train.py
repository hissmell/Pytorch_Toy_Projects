import torch
import numpy as np
from termcolor import colored
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


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

    print(colored(f"Epoch     : {epoch:4d}  (In {phase} session)",'yellow'))
    total_data_num = len(data_loader.dataset)
    running_loss = 0.0
    running_correct = 0
    criterion = CrossEntropyLoss(reduction='sum')
    for batch_idx,(inputs,labels) in enumerate(data_loader):
        if phase == 'training':
            optimizer.zero_grad()

        if phase == 'training':
            inputs.requires_grad = True
        else:
            inputs.requires_grad = False

        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = model(inputs)
        _,preds = torch.max(outputs.detach(),dim=1)
        # 여기에 loss 구현
        loss = criterion(outputs,labels)
        # 여기에 correct 구현
        correct = torch.sum(preds==labels)

        if phase == 'training':
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_correct += correct.item()
    torch.cuda.empty_cache()
    epoch_loss = running_loss / total_data_num
    epoch_accuracy = running_correct / total_data_num

    print(colored(f"Loss      : {epoch_loss:.6f}  Accuracy : {epoch_accuracy*100:4.8f}%",'blue'))
    return epoch_loss,epoch_accuracy

def record_training_data(training_data,phase,epoch_loss,epoch_accuracy):
    '''
    :param training_data:
    :param phase: 'training' or 'validation'
    :param epoch_loss:
    :param epoch_accuracy:
    :return:
    '''
    if phase == 'training':
        training_data['train_losses'].append(epoch_loss)
        training_data['train_accuracies'].append(epoch_accuracy)
    else:
        training_data['valid_losses'].append(epoch_loss)
        training_data['valid_accuracies'].append(epoch_accuracy)

