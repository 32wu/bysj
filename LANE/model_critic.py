# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import checkpoint_utils


# Model critic


def print_info(input_string=''):
    print('[94mMODEL_CRITIC_INFO|[0m', input_string)


class Critic(nn.Module):
    def __init__(self, input_size, output_size, small=False, optimizer_learning_rate=0.001, dev=torch.device('cpu')):
        super(Critic, self).__init__()
        self.dev = dev
        self.small = small
        self.grad_clip = None
        if self.small:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(64, output_size),)
        else:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(1024, output_size),)
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=optimizer_learning_rate)
        self.to(self.dev)

    def forward(self, x):
        output = self.fc3(self.fc2(self.fc1(x)))
        return output

    def set_grad_clip(self, grad_clip):
        self.grad_clip = grad_clip

    def set_learning_rate(self, learning_rate):
        self.optimizer_learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def learn(self, value_predict, value_target):
        difference = value_target - value_predict
        loss = (difference * difference).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

    def save_model(self, name=''):
        file_name = checkpoint_utils.resolve_checkpoint_file(name + '_1')
        torch.save(self.state_dict(), file_name)
        file_name = checkpoint_utils.resolve_checkpoint_file(name + '_2')
        torch.save(self.state_dict(), file_name)

    def load_model(self, name=''):
        try:
            file_name = checkpoint_utils.resolve_checkpoint_file(name + '_1')
            self.load_state_dict(torch.load(file_name))
        except Exception:
            print_info('Error: current1 model currupted.')
            file_name = checkpoint_utils.resolve_checkpoint_file(name + '_2')
            self.load_state_dict(torch.load(file_name))
        print_info('load: %s' % file_name)
