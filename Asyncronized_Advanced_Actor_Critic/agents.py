from models import SeparateActorCriticNetwork



class BasicAgent:
    def __init__(self,*args,**kwargs):
        self.action_space_size = None
        self.state_space_size = None
        self.action = None
        self.state = None
        self.model = None
        self.optimizer = None
        self.device = None
        pass

    def get_best_action_from_observation(self,observation):
        pass

    def get_best_action(self,state):
        pass

    def get_action_from_observation(self,observation):
        pass

    def get_action(self,state):
        pass

    def fit(self):
        pass

    def to(self,device = 'cpu'):
        self.device = device
        self.model.to(device)

