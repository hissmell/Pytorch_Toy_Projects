import os
import torch

def load_model_from(model_to,exp_name,model_name,check_point=None):
    model_path = os.path.join(os.getcwd(),exp_name, 'check_point_' + str(check_point), model_name + '.pth')
    model_to.load_state_dict(torch.load(model_path))
    model_to.eval()
    return model_to