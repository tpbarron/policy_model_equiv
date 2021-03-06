import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from running_stat import ObsNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CNNPolicy(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4) #, bias=False)
        # self.ab1 = AddBias(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) #, bias=False)
        # self.ab2 = AddBias(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1) #, bias=False)
        # self.ab3 = AddBias(32)

        self.linear1 = nn.Linear(32 * 7 * 7, 512) #, bias=False)
        # self.ab_fc1 = AddBias(512)

        self.critic_linear = nn.Linear(512, 1) #, bias=False)
        # self.ab_fc2 = AddBias(1)

        num_outputs = action_space.n
        self.actor_linear = nn.Linear(512, num_outputs) #, bias=False)
        # self.ab_fc3 = AddBias(num_outputs)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        self.train()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), self.actor_linear(x)

    def act(self, inputs):
        value, logits = self(inputs)
        probs = F.softmax(logits)
        action = probs.multinomial()
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"
        values, logits = self(inputs)
        log_probs = F.log_softmax(logits)
        probs = F.softmax(logits)
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return values, action_log_probs, dist_entropy



class CNNContinuousPolicy(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNContinuousPolicy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)
        self.critic_linear = nn.Linear(512, 1)

        num_outputs = action_space.shape[0]
        self.actor_linear = nn.Linear(512, num_outputs)
        self.a_log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        self.train()

    def encode(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        return x

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        value = self.critic_linear(x)
        action_mean = self.actor_linear(x)

        action_logstd = self.a_log_std.expand_as(action_mean)

        return value, action_mean, action_logstd


    def act(self, inputs, deterministic=False):
        value, action_mean, action_logstd = self(inputs)
        if deterministic:
            return value, action_mean
        # print ("value, actm, actlogstd:", value.size(), action_mean.size(), action_logstd.size())
        action_std = action_logstd.exp()
        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        action = action_mean + action_std * noise
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"
        value, action_mean, action_logstd = self(inputs)
        action_std = action_logstd.exp()
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return value, action_log_probs, dist_entropy

class CNNContinuousPolicySeparate(torch.nn.Module):
    def __init__(self, obs_shape, action_space):
        super(CNNContinuousPolicySeparate, self).__init__()

        num_inputs = obs_shape[0]
        d = obs_shape[-1]
        self.extra_conv = False

        if d == 32:
            self.actor_conv_reshape = 16 * 8 * 8
        elif d == 48:
            self.actor_conv_reshape = 16 * 12 * 12
            # self.actor_conv_reshape = 16 * 6 * 6
        elif d == 64:
            self.actor_conv_reshape = 16 * 8 * 8
        else:
            raise Exception

        self.conv1_a = nn.Conv2d(num_inputs, 16, 4, stride=2, padding=1)
        self.conv2_a = nn.Conv2d(16, 16, 4, stride=2, padding=1)
        if d > 48:
            self.conv3_a = nn.Conv2d(16, 16, 4, stride=2, padding=1)
            self.extra_conv = True
        self.linear1_a = nn.Linear(self.actor_conv_reshape, 32)
        self.fc_mean_a = nn.Linear(32, action_space.shape[0])
        self.a_log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        if d == 32:
            self.critic_conv_reshape = 16 * 16 * 16
        elif d == 48:
            self.critic_conv_reshape = 16 * 24 * 24
            # self.critic_conv_reshape = 16 * 12 * 12
        elif d == 64:
            self.critic_conv_reshape = 16 * 16 * 16
        else:
            raise Exception

        self.conv1_v = nn.Conv2d(num_inputs, 16, 4, stride=2, padding=1)
        if self.extra_conv:
            self.conv2_v = nn.Conv2d(16, 16, 4, stride=2, padding=1)
        self.linear1_v = nn.Linear(self.critic_conv_reshape, 32)
        self.enc_filter = ObsNorm((1, 32), clip=10.0)
        self.critic_linear_v = nn.Linear(32, 1)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1_a.weight.data.mul_(relu_gain)
        self.conv2_a.weight.data.mul_(relu_gain)
        self.linear1_a.weight.data.mul_(relu_gain)
        self.conv1_v.weight.data.mul_(relu_gain)
        self.linear1_v.weight.data.mul_(relu_gain)

        self.train()

    def encode(self, inputs):
        # print ("CNNContinuousPolicySeparate enc1")
        x = self.conv1_v(inputs / 255.0)
        x = F.relu(x)
        # print ("CNNContinuousPolicySeparate enc2")

        if self.extra_conv:
            x = self.conv2_v(x)
            x = F.relu(x)
        # print ("CNNContinuousPolicySeparate enc3")

        x = x.view(-1, self.critic_conv_reshape)
        x = self.linear1_v(x)
        # print ("CNNContinuousPolicySeparate enc4")
        x.data = self.enc_filter(x.data)
        # print ("CNNContinuousPolicySeparate enc5")
        return x

    def forward(self, inputs, encode_mean=False):
        d = inputs.size()[-1]

        x = self.conv1_v(inputs / 255.0)
        x = F.relu(x)

        if self.extra_conv:
            x = self.conv2_v(x)
            x = F.relu(x)

        # print (x.size())
        # input("")

        x = x.view(-1, self.critic_conv_reshape)
        x = self.linear1_v(x)

        if encode_mean:
            for i in range(x.size()[0]):
                self.enc_filter.update(x[i].data)

        x = F.relu(x)
        x = self.critic_linear_v(x)
        value = x

        x = self.conv1_a(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2_a(x)
        x = F.relu(x)

        if self.extra_conv:
            x = self.conv3_a(x)
            x = F.relu(x)

        # print (x.size())
        # input("")

        x = x.view(-1, self.actor_conv_reshape)
        x = self.linear1_a(x)
        x = F.relu(x)

        x = self.fc_mean_a(x)

        action_mean = x
        action_logstd = self.a_log_std.expand_as(action_mean)

        return value, action_mean, action_logstd

    def cuda(self, **args):
        super(CNNContinuousPolicySeparate, self).cuda(**args)
        self.enc_filter.cuda()

    def act(self, inputs, deterministic=False, encode_mean=False):
        value, action_mean, action_logstd = self(inputs, encode_mean=encode_mean)
        if deterministic:
            return value, action_mean
        # print ("value, actm, actlogstd:", value.size(), action_mean.size(), action_logstd.size())
        action_std = action_logstd.exp()
        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        action = action_mean + action_std * noise
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"
        value, action_mean, action_logstd = self(inputs)
        action_std = action_logstd.exp()
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return value, action_log_probs, dist_entropy


# class CNNContinuousPolicySeparate(torch.nn.Module):
#     def __init__(self, num_inputs, action_space):
#         super(CNNContinuousPolicySeparate, self).__init__()
#
#         self.conv_reshape = 16 * 3 * 3
#         self.conv1_a = nn.Conv2d(num_inputs, 32, 3, stride=2)
#         self.conv2_a = nn.Conv2d(32, 32, 3, stride=2)
#         self.conv3_a = nn.Conv2d(32, 16, 3, stride=2)
#         self.linear1_a = nn.Linear(self.conv_reshape, 64)
#         self.fc_mean_a = nn.Linear(64, action_space.shape[0])
#         self.a_log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))
#
#         self.conv1_v = nn.Conv2d(num_inputs, 32, 3, stride=2)
#         self.conv2_v = nn.Conv2d(32, 32, 3, stride=2)
#         self.conv3_v = nn.Conv2d(32, 16, 3, stride=2)
#         self.linear1_v = nn.Linear(self.conv_reshape, 64)
#         self.enc_filter = ObsNorm((1, 64), clip=10.0)
#
#         self.critic_linear_v = nn.Linear(64, 1)
#
#         self.apply(weights_init)
#
#         relu_gain = nn.init.calculate_gain('tanh')
#         self.conv1_a.weight.data.mul_(relu_gain)
#         self.conv2_a.weight.data.mul_(relu_gain)
#         self.conv3_a.weight.data.mul_(relu_gain)
#         self.linear1_a.weight.data.mul_(relu_gain)
#         self.conv1_v.weight.data.mul_(relu_gain)
#         self.conv2_v.weight.data.mul_(relu_gain)
#         self.conv3_v.weight.data.mul_(relu_gain)
#         self.linear1_v.weight.data.mul_(relu_gain)
#
#         self.train()
#
#     def encode(self, inputs):
#         x = self.conv1_v(inputs / 255.0)
#         x = F.tanh(x)
#
#         x = self.conv2_v(x)
#         x = F.tanh(x)
#
#         x = self.conv3_v(x)
#         x = F.tanh(x)
#
#         x = x.view(-1, self.conv_reshape)
#         x = self.linear1_v(x)
#         x.data = self.enc_filter(x.data)
#         return x
#
#     def forward(self, inputs, encode_mean=False):
#         x = self.conv1_v(inputs / 255.0)
#         x = F.tanh(x)
#
#         x = self.conv2_v(x)
#         x = F.tanh(x)
#
#         x = self.conv3_v(x)
#         x = F.tanh(x)
#
#         # print (x.size())
#         # input("")
#
#         x = x.view(-1, self.conv_reshape)
#         x = self.linear1_v(x)
#
#         if encode_mean:
#             for i in range(x.size()[0]):
#                 self.enc_filter.update(x[i].data)
#
#         x = F.tanh(x)
#         x = self.critic_linear_v(x)
#         value = x
#
#         x = self.conv1_a(inputs / 255.0)
#         x = F.tanh(x)
#
#         x = self.conv2_a(x)
#         x = F.tanh(x)
#
#         x = self.conv3_a(x)
#         x = F.tanh(x)
#
#         # print (x.size())
#         # input("")
#
#         x = x.view(-1, self.conv_reshape)
#         x = self.linear1_a(x)
#         x = F.tanh(x)
#
#         x = self.fc_mean_a(x)
#
#         action_mean = x
#         action_logstd = self.a_log_std.expand_as(action_mean)
#
#         return value, action_mean, action_logstd
#
#     def cuda(self, **args):
#         super(CNNContinuousPolicySeparate, self).cuda(**args)
#         self.enc_filter.cuda()
#
#     def act(self, inputs, deterministic=False, encode_mean=False):
#         value, action_mean, action_logstd = self(inputs, encode_mean=encode_mean)
#         if deterministic:
#             return value, action_mean
#         # print ("value, actm, actlogstd:", value.size(), action_mean.size(), action_logstd.size())
#         action_std = action_logstd.exp()
#         noise = Variable(torch.randn(action_std.size()))
#         if action_std.is_cuda:
#             noise = noise.cuda()
#
#         action = action_mean + action_std * noise
#         return value, action
#
#     def evaluate_actions(self, inputs, actions):
#         assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"
#         value, action_mean, action_logstd = self(inputs)
#         action_std = action_logstd.exp()
#         action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
#         action_log_probs = action_log_probs.sum(1, keepdim=True)
#         dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
#         dist_entropy = dist_entropy.sum(-1).mean()
#         return value, action_log_probs, dist_entropy

class CNN3ContinuousPolicySeparate(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNN3ContinuousPolicySeparate, self).__init__()

        self.conv_reshape = 16 * 3 * 3
        self.conv1_a = nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=(1, 0, 0))
        self.conv2_a = nn.Conv3d(32, 32, 3, stride=(1, 2, 2), padding=(1, 0, 0))
        self.conv3_a = nn.Conv3d(32, 16, (4, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.linear1_a = nn.Linear(self.conv_reshape, 64)
        self.fc_mean_a = nn.Linear(64, action_space.shape[0])
        self.a_log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        self.conv1_v = nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=(1, 0, 0))
        self.conv2_v = nn.Conv3d(32, 32, 3, stride=(1, 2, 2), padding=(1, 0, 0))
        self.conv3_v = nn.Conv3d(32, 16, (4, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.linear1_v = nn.Linear(self.conv_reshape, 64)
        self.enc_filter = ObsNorm((1, 64), clip=10.0)

        self.critic_linear_v = nn.Linear(64, 1)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('tanh')
        self.conv1_a.weight.data.mul_(relu_gain)
        self.conv2_a.weight.data.mul_(relu_gain)
        self.conv3_a.weight.data.mul_(relu_gain)
        self.linear1_a.weight.data.mul_(relu_gain)
        self.conv1_v.weight.data.mul_(relu_gain)
        self.conv2_v.weight.data.mul_(relu_gain)
        self.conv3_v.weight.data.mul_(relu_gain)
        self.linear1_v.weight.data.mul_(relu_gain)

        self.train()

    def encode(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, 4, 32, 32)
        x = self.conv1_v(inputs / 255.0)
        x = F.tanh(x)

        x = self.conv2_v(x)
        x = F.tanh(x)

        x = self.conv3_v(x)
        x = F.tanh(x)

        x = x.view(-1, self.conv_reshape)
        x = self.linear1_v(x)
        x.data = self.enc_filter(x.data)
        return x

    def forward(self, inputs, encode_mean=False):
        # print (inputs.size())
        inputs = inputs.view(inputs.size()[0], 1, 4, 32, 32)
        # inputs = inputs.view(4, 1, 4, 32, 32)
        x = self.conv1_v(inputs / 255.0)
        x = F.tanh(x)

        x = self.conv2_v(x)
        x = F.tanh(x)

        x = self.conv3_v(x)
        x = F.tanh(x)
        # print (x.size())
        x = x.view(-1, self.conv_reshape)
        # print (x.size())
        # input("")
        x = self.linear1_v(x)

        if encode_mean:
            for i in range(x.size()[0]):
                self.enc_filter.update(x[i].data)

        x = F.tanh(x)
        x = self.critic_linear_v(x)
        value = x

        x = self.conv1_a(inputs / 255.0)
        x = F.tanh(x)

        x = self.conv2_a(x)
        x = F.tanh(x)

        x = self.conv3_a(x)
        x = F.tanh(x)

        # print (x.size())
        # input("")

        x = x.view(-1, self.conv_reshape)
        x = self.linear1_a(x)
        x = F.tanh(x)

        x = self.fc_mean_a(x)

        action_mean = x
        action_logstd = self.a_log_std.expand_as(action_mean)

        # print (value.size(), action_mean.size(), action_logstd.size())
        return value, action_mean, action_logstd

    def cuda(self, **args):
        super(CNN3ContinuousPolicySeparate, self).cuda(**args)
        self.enc_filter.cuda()

    def act(self, inputs, deterministic=False, encode_mean=False):
        value, action_mean, action_logstd = self(inputs, encode_mean=encode_mean)
        if deterministic:
            return value, action_mean
        # print ("value, actm, actlogstd:", value.size(), action_mean.size(), action_logstd.size())
        action_std = action_logstd.exp()
        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        action = action_mean + action_std * noise
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"
        value, action_mean, action_logstd = self(inputs)
        action_std = action_logstd.exp()
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return value, action_log_probs, dist_entropy


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class LatentModel(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(LatentModel, self).__init__()
        self.enc_filter = ObsNorm((1, 32), clip=10.0)
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.fcr = nn.Linear(num_inputs, 1) # reward est
        self.apply(weights_init_mlp)

        self.train()

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.enc_filter.cuda()

    def forward(self, inputs):
        # TODO: for now try without filter because it will kill gradients
        # inputs.data = self.obs_filter(inputs.data)
        x = self.fc1(inputs)
        r = self.fcr(inputs)
        return x, r

class MLPPolicy(torch.nn.Module):
    def __init__(self, num_inputs, action_space, do_encode_mean=True):
        print ("Making shared MLP actor critic!")
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))
        self.fc_val = nn.Linear(64, 1)
        self.latent_model = LatentModel(64+action_space.shape[0], 64)

        self.apply(weights_init_mlp)

        tanh_gain = nn.init.calculate_gain('tanh')
        self.fc_mean.weight.data.mul_(0.01)

        self.train()

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def forward(self, inputs, encode_mean=False):
        self.obs_filter.update(inputs.data)
        inputs.data = self.obs_filter(inputs.data)

        x = F.tanh(self.fc1(inputs))
        x = F.tanh(self.fc2(x))
        enc = x
        value = self.fc_val(x)
        action_mean = self.fc_mean(x)
        action_logstd = self.log_std.expand_as(action_mean)

        return value, action_mean, action_logstd, enc

    def act(self, inputs, deterministic=False, encode_mean=False):
        value, action_mean, action_logstd, enc = self(inputs, encode_mean=encode_mean)
        if deterministic:
            return value, action_mean
        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        action = action_mean + action_std * noise
        # print ("Act: ", action.size(), enc.size())
        mpred, rpred = self.latent_model(torch.cat((action, enc), dim=1))
        return value, action, enc, mpred, rpred

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 2, "Expect to have inputs in num_processes * num_steps x ... format"

        # print ("Inputs: ", inputs.size(), actions.size())

        value, action_mean, action_logstd, enc = self(inputs)
        mpred, rpred = self.latent_model(torch.cat((actions, enc), dim=1))
        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return value, action_log_probs, dist_entropy, mpred, rpred

class MLPPolicySeparate(torch.nn.Module):
    def __init__(self, num_inputs, action_space, do_encode_mean=True):
        super(MLPPolicySeparate, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)
        self.a_fc_mean = nn.Linear(64, action_space.shape[0])
        self.a_log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        self.v_fc1 = nn.Linear(num_inputs, 32)
        self.v_fc2 = nn.Linear(32, 32)
        self.v_fc3 = nn.Linear(32, 1)

        self.latent_model = LatentModel(32+action_space.shape[0], 32)

        self.apply(weights_init_mlp)

        self.do_encode_mean = do_encode_mean

        tanh_gain = nn.init.calculate_gain('tanh')
        #self.a_fc1.weight.data.mul_(tanh_gain)
        #self.a_fc2.weight.data.mul_(tanh_gain)
        self.a_fc_mean.weight.data.mul_(0.01)
        #self.v_fc1.weight.data.mul_(tanh_gain)
        #self.v_fc2.weight.data.mul_(tanh_gain)

        self.train()

    def cuda(self, **args):
        super(MLPPolicySeparate, self).cuda(**args)
        self.obs_filter.cuda()

    def forward(self, inputs, encode_mean=False):
        self.obs_filter.update(inputs.data)
        inputs.data = self.obs_filter(inputs.data)
        x = self.v_fc1(inputs)
        x = F.tanh(x)
        x = self.v_fc2(x)
        x = F.tanh(x)
        enc = x
        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        x = self.a_fc_mean(x)
        action_mean = x

        action_logstd = self.a_log_std.expand_as(action_mean)
        return value, action_mean, action_logstd, enc

    def act(self, inputs, deterministic=False, encode_mean=False):
        value, action_mean, action_logstd, enc = self(inputs, encode_mean=encode_mean)
        if deterministic:
            return value, action_mean
        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        action = action_mean + action_std * noise
        # print ("Act: ", action.size(), enc.size())
        mpred, rpred = self.latent_model(torch.cat((action, enc), dim=1))
        return value, action, enc, mpred, rpred

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 2, "Expect to have inputs in num_processes * num_steps x ... format"

        # print ("Inputs: ", inputs.size(), actions.size())

        value, action_mean, action_logstd, enc = self(inputs)
        mpred, rpred = self.latent_model(torch.cat((actions, enc), dim=1))
        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return value, action_log_probs, dist_entropy, mpred, rpred

def make_actor_critic(observation_space, action_space, is_shared, is_continuous, is_encode_mean=True):
    # actor_critic = CNN3ContinuousPolicySeparate(observation_space[0], action_space)
    # return actor_critic
    if not is_continuous:
        if len(observation_space) > 1: # then use conv
            actor_critic = CNNPolicy(observation_space[0], action_space)
        else:
            raise NotImplementedError
    else:
        if len(observation_space) > 1: # use conv
            if is_shared:
                actor_critic = CNNContinuousPolicy(observation_space[0], action_space)
            else:
                actor_critic = CNNContinuousPolicySeparate(observation_space, action_space)
        else:
            if is_shared:
                actor_critic = MLPPolicy(observation_space[0], action_space, do_encode_mean=is_encode_mean)
            else:
                actor_critic = MLPPolicySeparate(observation_space[0], action_space, do_encode_mean=is_encode_mean)
    return actor_critic
