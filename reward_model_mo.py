from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
import loralib as lora

from scipy.stats import norm

device = 'cpu'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh', use_lora=False, rank=16, lora_alpha=16):
    net = []
    if not use_lora:
        for i in range(n_layers):
            net.append(nn.Linear(in_size, H))
            net.append(nn.LeakyReLU())
            in_size = H
        net.append(nn.Linear(in_size, out_size))
    else:
        for i in range(n_layers):
            net.append(lora.Linear(in_size, H, r=rank, lora_alpha=lora_alpha))
            net.append(nn.LeakyReLU())
            in_size = H
        net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 pretrained_model=False,
                 use_lora=False,
                 rank=16,
                 model_name=None,
                 work_dir=None,
                 lora_alpha=16, surf=False, real=False,
                 multi_obj=False, weighted=False, polite=False, env_name="default"):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.multi_obj = multi_obj
        self.weighted = weighted
        self.polite = polite
        self.env_name = env_name
        
        self.capacity = int(capacity)
        self.buffer_seg1 = []
        self.buffer_seg2 = []
        self.buffer_label = []
        self.buffer_index = []
        self.highlights_max = [[] for i in range(ensemble_size)]
        self.highlights_min = [[] for i in range(ensemble_size)]
        for i in range(ensemble_size):
            self.buffer_seg1.append(np.empty((self.capacity, 50, self.ds+self.da), dtype=np.float32))
            self.buffer_seg2.append(np.empty((self.capacity, 50, self.ds+self.da), dtype=np.float32))
            self.buffer_label.append(np.empty((self.capacity, 1), dtype=np.float32))
            self.buffer_index.append(0)


        self.buffer_full = False

        self.use_lora = use_lora
        self.lora_alpha = lora_alpha
        self.rank = rank
        self.construct_ensemble()

        #Load pretrained function
        if pretrained_model and polite:
            self.load2("/root/code/rl_zoo/Policy/window_open", "mosaic_reward_model_1000000")
        if pretrained_model and not polite:
            self.load2("/root/code/rl_zoo/Policy/window_open", "base_reward_model_1000000")



        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation, use_lora=self.use_lora, rank=self.rank, lora_alpha=self.lora_alpha)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rews, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rews
        dims = len(rews)


        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, dims)

        init_data = len(self.inputs) == 0

        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # the network parameterizes r hat in eqn 1 from the paper
        if self.weighted:
            if self.env_name == "mo-hopper-v4":
                if member == 0:
                    return F.softmax(r_hat, dim=-1)[:,0] * 1.0
                elif member == 1:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.7
                elif member == 2:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.2
                elif member == 3:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.1
                else:
                    return F.softmax(r_hat, dim=-1)[:,0] * 1.0
            elif self.env_name == "mo-halfcheetah-v4":
                if member == 0:
                    return F.softmax(r_hat, dim=-1)[:,0] * 1
                elif member == 1:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.8
                elif member == 2:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.2
                else:
                    return F.softmax(r_hat, dim=-1)[:,0] * 1.0
            elif self.env_name == "metaworld_drawer-close-v2" or self.env_name == "metaworld_button-press-v2" or self.env_name == "metaworld_button-press-wall-v2" or self.env_name == "metaworld_window-open-v2":
                if member == 0:
                    return F.softmax(r_hat, dim=-1)[:,0] * 1
                elif member == 1:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.8
                elif member == 2:
                    return F.softmax(r_hat, dim=-1)[:,0] * 0.2
                else:
                    return F.softmax(r_hat, dim=-1)[:,0]
            else:
                return F.softmax(r_hat, dim=-1)[:,0]
        else:
            return F.softmax(r_hat, dim=-1)[:,0]
        # taking 0 index for probability x_1 > x_2
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member_old(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        if self.weighted:
            if self.env_name == "mo-hopper-v4":
                if member == 0:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 1
                elif member == 1:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.7
                elif member == 2:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.2
                elif member == 3:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.1
                else:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device))
            elif self.env_name == "mo-halfcheetah-v4":
                if member == 0:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 1
                elif member == 1:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.8
                elif member == 2:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.2
                else:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device))
            elif self.env_name == "metaworld_drawer-close-v2" or self.env_name == "metaworld_button-press-v2" or self.env_name == "metaworld_button-press-wall-v2" or self.env_name == "metaworld_window-open-v2":
                if member == 0:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 1
                elif member == 1:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.8
                elif member == 2:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device)) * 0.2
                else:
                    return self.ensemble[member](torch.from_numpy(x).float().to(device))
            else:
                return self.ensemble[member](torch.from_numpy(x).float().to(device))
        else:
            return self.ensemble[member](torch.from_numpy(x).float().to(device))


    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_all(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return r_hats
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    def load2(self, model_dir, name, load_lora=False):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/%s_%s.pt' % (model_dir, name, member)), strict=False
            )
            if load_lora:

                self.ensemble[member].load_state_dict(
                    torch.load('%s/ckpt_lora_%s.pt' % (model_dir, member)), strict=False
                )

    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x dim of r

        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x dim of r

        r_t_1_mo = []
        r_t_2_mo = []
        print("CHECKKING")
        print(train_inputs.shape)
        print(train_targets.shape)
        print(sa_t_1.shape)
        print(r_t_1.shape)
        for i in range(self.de):
            r_t_1_mo.append(r_t_1[:, :, i:i + 1])
            r_t_2_mo.append(r_t_2[:, :, i:i + 1])



        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        #r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        #r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        for i in range(self.de):
            print(r_t_1_mo[i].shape)
            r_t_1_mo[i] = r_t_1_mo[i].reshape(-1, r_t_1_mo[i].shape[-1])
            print(r_t_1_mo[i].shape)
            r_t_2_mo[i] = r_t_2_mo[i].reshape(-1, r_t_2_mo[i].shape[-1])

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a

        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        for i in range(self.de):
            r_t_1_mo[i] = np.take(r_t_1_mo[i], time_index_1, axis=0)  # Batch x size_seg x 1
            r_t_2_mo[i] = np.take(r_t_2_mo[i], time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1_mo, r_t_2_mo

    def put_queries(self, sa_t_1, sa_t_2, labels, index):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index[index] + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index[index]
            np.copyto(self.buffer_seg1[index][self.buffer_index[index]:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[index][self.buffer_index[index]:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[index][self.buffer_index[index]:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[index][0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[index][0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[index][0:remain], labels[maximum_index:])

            self.buffer_index[index] = remain
        else:
            np.copyto(self.buffer_seg1[index][self.buffer_index[index]:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[index][self.buffer_index[index]:next_index], sa_t_2)
            np.copyto(self.buffer_label[index][self.buffer_index[index]:next_index], labels)
            self.buffer_index[index] = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    def get_label_with_highlights_metaworld(self, sa_t_1, sa_t_2, r_t_1, r_t_2, k=5, member=0):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = int(r_t_1.shape[1])
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels


        distances1 = []
        distances2 = []
        for x in range(sa_t_1.shape[0]):
            traj_dist1 = []
            traj_dist2 = []
            for y in range(sa_t_1.shape[1]):
                hand_pos = [sa_t_1[x][y][0], sa_t_1[x][y][1], sa_t_1[x][y][2]]
                obj_pos = [sa_t_1[x][y][4], sa_t_1[x][y][5], sa_t_1[x][y][6]]
                distance = math.sqrt(
            (obj_pos[0] - hand_pos[0]) ** 2 + (obj_pos[1] - hand_pos[1]) ** 2 + (obj_pos[2] - hand_pos[2]) ** 2)
                traj_dist1.append(distance)

                hand_pos = [sa_t_2[x][y][0], sa_t_2[x][y][1], sa_t_2[x][y][2]]
                obj_pos = [sa_t_2[x][y][4], sa_t_2[x][y][5], sa_t_2[x][y][6]]
                distance = math.sqrt(
                    (obj_pos[0] - hand_pos[0]) ** 2 + (obj_pos[1] - hand_pos[1]) ** 2 + (obj_pos[2] - hand_pos[2]) ** 2)
                traj_dist2.append(distance)
            distances1.append(traj_dist1)
            distances2.append(traj_dist2)

        if member == 1 or member == 0:
            specific_state_values_1 = np.array(distances1)
            specific_state_values_2 = np.array(distances2)
            print(specific_state_values_1.shape)
            top_k_indices_1 = []
            top_k_indices_2 = []

            for batch in range(specific_state_values_1.shape[0]):
                # Use argsort to sort the array and pick the indices of the top 5 lowest values
                indices = np.argsort(specific_state_values_1[batch])[-k:]
                top_k_indices_1.append(indices)
                indices = np.argsort(specific_state_values_2[batch])[-k:]
                top_k_indices_2.append(indices)

            # max_values, max_indices = torch.topk(specific_state_values, 5, dim=1)  # Find the top indices for each example

            print(top_k_indices_1)
            print(seg_size)
            for i in range(len(top_k_indices_1)):
                if labels[i] == 0:
                    fish = [1.0 if j in top_k_indices_1[i] else 0.0 for j in range(seg_size)]
                    self.highlights_max[member].append(fish)
                elif labels[i] == 1:
                    fish = [1.0 if j in top_k_indices_2[i] else 0.0 for j in range(seg_size)]
                    self.highlights_max[member].append(fish)

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels


    def get_label_with_highlights(self, sa_t_1, sa_t_2, r_t_1, r_t_2, state_value_index=0, k=5, member=0):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels


        if state_value_index != -1:
            specific_state_values_1 = sa_t_1[:, :, state_value_index]  # Extract the specific state values from each state
            specific_state_values_2 = sa_t_2[:, :, state_value_index]  # Extract the specific state values from each state

            #max_values, max_indices = torch.topk(specific_state_values, 5, dim=1)  # Find the top indices for each example

            partition_indices_1 = np.argpartition(-specific_state_values_1, k, axis=1)[:, :k]
            partition_indices_2 = np.argpartition(-specific_state_values_2, k, axis=1)[:, :k]
            top_k_indices_1 = np.array([arr[np.argsort(-specific_state_values_1[i, arr])] for i, arr in enumerate(partition_indices_1)])
            top_k_indices_2 = np.array([arr[np.argsort(-specific_state_values_2[i, arr])] for i, arr in enumerate(partition_indices_2)])
            for i in range(top_k_indices_1.shape[0]):
                if labels[i] == 0:
                    fish = [1.0 if j in top_k_indices_1[i] else 0.0 for j in range(seg_size)]
                    self.highlights_max[member].append(fish)
                elif labels[i] == 1:
                    fish = [1.0 if j in top_k_indices_2[i] else 0.0 for j in range(seg_size)]
                    self.highlights_max[member].append(fish)

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    def get_label_surf(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
        ent = F.softmax(r_hat, dim=-1)[:, 1]
        print(ent[:5])
        print(r_hat[:5])
        label_indexes = (ent > 0.99).reshape(-1)
        print(label_indexes[:5])
        print("label_indexes")
        print(len(label_indexes))
        print("ent:")
        print(len(ent))
        ent = ent[label_indexes]
        print("new ent:")
        print(len(ent))
        sa_t_1 = sa_t_1[label_indexes]
        sa_t_2 = sa_t_2[label_indexes]
        r_t_1 = r_t_1[label_indexes]
        r_t_2 = r_t_2[label_indexes]
        labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        print(ent[:5])
        print(labels[:5])


        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    def kcenter_sampling(self):
        
        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1[0], r_t_2[0])
        n_alt_prefs = 10
        if self.multi_obj:
            if len(labels) > 0:
                self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, 0)

            for i in range(1, self.de):
                state_value_index = 0
                if self.env_name == "metaworld_drawer-close-v2" or self.env_name == "metaworld_button-press-v2" or self.env_name == "metaworld_button-press-wall-v2" or self.env_name == "metaworld_window-open-v2" and i == 1:
                    tmp_r_t_1 = np.sum(r_t_1[i], axis=(1, 2))
                    top_k_index = (-tmp_r_t_1).argsort()[:n_alt_prefs]
                    sa_t_1_tmp = sa_t_1[top_k_index]
                    sa_t_2_tmp = sa_t_2[top_k_index]
                    r_t_1_tmp = r_t_1[i][top_k_index]
                    r_t_2_tmp = r_t_2[i][top_k_index]

                    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label_with_highlights_metaworld(
                        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, k=n_alt_prefs, member=i)
                    if len(labels) > 0:
                        self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)
                else:
                    if self.env_name == "mo-hopper-v4" and i == 1:
                        state_value_index = 5
                    elif self.env_name == "mo-hopper-v4" and i == 2:
                        state_value_index = 0
                    elif self.env_name == "mo-halfcheetah-v4" and i == 1:
                        state_value_index = 8
                    else:
                        state_value_index = -1

                    tmp_r_t_1 = np.sum(r_t_1[i], axis=(1, 2))

                    top_k_index = (-tmp_r_t_1).argsort()[:n_alt_prefs]
                    sa_t_1_tmp = sa_t_1[top_k_index]
                    sa_t_2_tmp = sa_t_2[top_k_index]
                    r_t_1_tmp = r_t_1[i][top_k_index]
                    r_t_2_tmp = r_t_2[i][top_k_index]

                    # sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label(
                    #    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp)
                    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label_with_highlights(
                        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, state_value_index, k=n_alt_prefs, member=i)
                    if len(labels) > 0:
                        self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)
        else:
            if len(labels) > 0:
                for i in range(1, self.de):
                    self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)

        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        sa_t_1 = sa_t_1[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        for i in range(self.de):
            r_t_1[i] = r_t_1[i][top_k_index]
            r_t_2[i] = r_t_2[i][top_k_index]
            print(r_t_1[i].shape)

        # get labels
        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1[0], r_t_2[0])
        n_alt_prefs = 25
        if self.multi_obj:
            if len(labels) > 0:
                self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, 0)

            for i in range(1, self.de):
                state_value_index = 0
                if self.env_name == "metaworld_drawer-close-v2" or self.env_name == "metaworld_button-press-v2" or self.env_name == "metaworld_button-press-wall-v2" or self.env_name == "metaworld_window-open-v2" and i == 1:
                    tmp_r_t_1 = np.sum(r_t_1[i], axis=(1, 2))
                    top_k_index = (-tmp_r_t_1).argsort()[:n_alt_prefs]
                    sa_t_1_tmp = sa_t_1[top_k_index]
                    sa_t_2_tmp = sa_t_2[top_k_index]
                    r_t_1_tmp = r_t_1[i][top_k_index]
                    r_t_2_tmp = r_t_2[i][top_k_index]

                    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label_with_highlights_metaworld(
                        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, k=n_alt_prefs, member=i)
                    if len(labels) > 0:
                        self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)
                else:
                    if self.env_name == "mo-hopper-v4" and i == 1:
                        state_value_index = 5
                    elif self.env_name == "mo-hopper-v4" and i == 2:
                        state_value_index = 0
                    elif self.env_name == "mo-halfcheetah-v4" and i == 1:
                        state_value_index = 8
                    else:
                        state_value_index = -1

                    tmp_r_t_1 = np.sum(r_t_1[i], axis=(1, 2))

                    top_k_index = (-tmp_r_t_1).argsort()[:n_alt_prefs]
                    sa_t_1_tmp = sa_t_1[top_k_index]
                    sa_t_2_tmp = sa_t_2[top_k_index]
                    r_t_1_tmp = r_t_1[i][top_k_index]
                    r_t_2_tmp = r_t_2[i][top_k_index]

                    #sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label(
                    #    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp)
                    sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, labels = self.get_label_with_highlights(
                        sa_t_1_tmp, sa_t_2_tmp, r_t_1_tmp, r_t_2_tmp, state_value_index, k=n_alt_prefs, member=i)
                    if len(labels) > 0:
                        self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)
        else:
            if len(labels) > 0:
                for i in range(1, self.de):
                    self.put_queries(sa_t_1_tmp, sa_t_2_tmp, labels, i)


        return len(labels)

    def disagreement_sampling_surf(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        sorted_index = (-disagree).argsort()
        top_k_index = sorted_index[:self.mb_size]

        # SURF
        unlabeled_index = sorted_index[self.mb_size:self.mb_size + self.mb_size * 4]
        r_t_1_unlabeled, sa_t_1_unlabeled = r_t_1[unlabeled_index], sa_t_1[unlabeled_index]
        r_t_2_unlabeled, sa_t_2_unlabeled = r_t_2[unlabeled_index], sa_t_2[unlabeled_index]

        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]


        print("Number of unlabeled:")
        print(len(unlabeled_index))

        # SURF
        sa_t_1_unlabeled, sa_t_2_unlabeled, r_t_1_unlabeled, r_t_2_unlabeled, labels_unlabeled = self.get_label_surf(
            sa_t_1_unlabeled, sa_t_2_unlabeled, r_t_1_unlabeled, r_t_2_unlabeled)
        print("Number of unlabeled that got labeled:")
        print(len(labels_unlabeled))
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        new_sa_t_1 = []
        new_sa_t_2 = []
        for i in range(len(sa_t_1)):
            H = self.size_segment
            H_tmp = 50
            k_1 = np.random.randint(0, H - H_tmp-1)
            k_2 = np.random.randint(0, H - H_tmp-1)
            new_sa_t_1.append(sa_t_1[i][k_1:k_1 + H_tmp])
            new_sa_t_2.append(sa_t_2[i][k_2:k_2 + H_tmp])
            if i < 2:
                print(H_tmp)
                print(k_1)
        sa_t_1 = np.array(new_sa_t_1, dtype=np.float32)
        sa_t_2 = np.array(new_sa_t_2, dtype=np.float32)

        new_sa_t_1_unlabeled = []
        new_sa_t_2_unlabeled = []
        for i in range(len(sa_t_1_unlabeled)):
            H = self.size_segment
            H_tmp = 50
            k_1 = np.random.randint(0, H - H_tmp-1)
            k_2 = np.random.randint(0, H - H_tmp-1)
            new_sa_t_1_unlabeled.append(sa_t_1_unlabeled[i][k_1:k_1 + H_tmp])
            new_sa_t_2_unlabeled.append(sa_t_2_unlabeled[i][k_2:k_2 + H_tmp])
            if i < 2:
                print(H_tmp)
                print(k_1)
        sa_t_1_unlabeled = np.array(new_sa_t_1_unlabeled, dtype=np.float32)
        sa_t_2_unlabeled = np.array(new_sa_t_2_unlabeled, dtype=np.float32)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        if len(labels_unlabeled) > 0:
            self.put_queries(sa_t_1_unlabeled, sa_t_2_unlabeled, labels_unlabeled)

        return len(labels)
    
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)

    def get_critical_points_rewards(self, hl_pos, prefs, r1_rolled, r2_rolled):
        critical_points_discounted_reward_punishment = torch.zeros_like(r1_rolled)
        critical_points_discounted_reward_approve = torch.zeros_like(r1_rolled)
        for i in range(len(prefs)):
            if prefs[i] == 0:
                #critical_points_discounted_reward_punishment[i] = r1_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r1_rolled[i] * torch.tensor(hl_pos[i], dtype=torch.float32).unsqueeze(-1)
            if prefs[i] == 1:
                #critical_points_discounted_reward_punishment[i] = r2_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r2_rolled[i] * torch.tensor(hl_pos[i], dtype=torch.float32).unsqueeze(-1)

        #punishments_in_batch = torch.sum(critical_points[:, :, 0] == 1).item()
        #approvements_in_batch = torch.sum(critical_points[:, :, 1] == 1).item()

        #punishment_reward = torch.sum(critical_points_discounted_reward_punishment)  # / punishments_in_batch
        approve_reward = torch.sum(critical_points_discounted_reward_approve)  # / approvements_in_batch
        return approve_reward, 0, 1, 1


    def train_reward(self, num_epochs=None):
        if self.use_lora:
            for member in self.ensemble:
                lora.mark_only_lora_as_trainable(member)

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index[0]
        total_batch_index = []
        for i in range(self.de):
            max_len = self.capacity if self.buffer_full else self.buffer_index[i]
            total_batch_index.append(np.random.permutation(max_len))
        max_len = self.capacity if self.buffer_full else self.buffer_index[0]
        if not num_epochs:
            num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = [0 for _ in range(self.de)]

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[member][idxs]
                sa_t_2 = self.buffer_seg2[member][idxs]
                labels = self.buffer_label[member][idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                total[member] += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)

                approve_reward, punishment_reward = 0, 0
                if self.polite:
                    hl = []
                    if len(self.highlights_max[member]) == 1:
                        hl_pos = [self.highlights_max[member][i] for i in idxs]
                        approve_reward, punishment_reward, n_approve, n_punishment = self.get_critical_points_rewards(hl_pos, labels, r_hat1, r_hat2)
                        approve_reward = approve_reward / n_approve
                        punishment_reward = punishment_reward / n_punishment


                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                if self.polite:
                    loss += (curr_loss - approve_reward)
                else:
                    loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc

    def train_reward_bagging(self, num_epochs=None):
        if self.use_lora:
            for member in self.ensemble:
                lora.mark_only_lora_as_trainable(member)

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        if not num_epochs:
            num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def get_preference_probability(self, queries):
        probabilities = []

        for query in queries:
            trajectory_1, trajectory_2 = query

            # Convert lists to numpy arrays
            trajectory_1_np = np.array(trajectory_1)
            trajectory_2_np = np.array(trajectory_2)

            # Compute r_hat sums for the entire trajectory
            # Sum the predicted rewards for all action-state pairs in the trajectory
            r_hat1_total = self.r_hat_member(trajectory_1_np).sum().detach().numpy()  # Sum and detach
            r_hat2_total = self.r_hat_member(trajectory_2_np).sum().detach().numpy()  # Sum and detach

            # Use the total r_hat values for each trajectory to compute the softmax preference
            r_hat_combined = np.array([[r_hat1_total, r_hat2_total]])

            # Convert back to torch tensor for softmax
            preferences = F.softmax(torch.from_numpy(r_hat_combined).float(), dim=-1)

            # Convert preferences back to NumPy array if needed
            probabilities.append(preferences.detach().numpy()[0])  # Use [0] to get the first (and only) row as output
        return probabilities

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[member][idxs]
                sa_t_2 = self.buffer_seg2[member][idxs]
                labels = self.buffer_label[member][idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc