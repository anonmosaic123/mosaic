#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
#from reward_model import RewardModel
from reward_model_mo import RewardModel
#from reward_model_bagging import RewardModel
from collections import deque

import utils
import hydra
import wandb

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.pretrained_model = cfg.pretrained_model
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.model_name = cfg.model_name
        self.env_name = cfg.env

        self.multi_obj = self.cfg.multi_obj

        """WANDB LOGGING UNCOMMENT IF YOU WANT TO UTILIZE W&B"""
        """
        wandb.login(key="XYZ")
        run_name = f"{self.cfg.env}__multi-{self.cfg.multi_obj}__polite-{self.cfg.polite}__weighted-{self.cfg.weighted}__fb-{self.cfg.max_feedback}__seed{self.cfg.seed}"
        config = {"n_queries": cfg.max_feedback,
                  "env": cfg.env,
                  "multi_obj": cfg.multi_obj,
                  "weighted": cfg.weighted,
                  "polite": cfg.polite}
        run = wandb.init(
            name=run_name,
            project="PrefLearn",
            entity="sholk",
            config=config,
            sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        if wandb.run is not None:
            wandb.define_metric("reward/step")
            wandb.define_metric("reward/*", step_metric="reward/step")
        """

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            pretrained_model=self.pretrained_model,
            model_name=self.model_name,
            work_dir=self.work_dir,
            multi_obj=self.multi_obj,
            weighted=self.cfg.weighted,
            polite=self.cfg.polite,
            env_name=self.env_name
        )
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            if self.env_name == "mo-halfcheetah-v4" or self.env_name == "mo-reacher-v4" or self.env_name == "mo-hopper-v4":
                obs, info = self.env.reset()
            else:
                obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            true_obj_episode_reward = [0, 0]
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                if self.env_name == "mo-halfcheetah-v4" or self.env_name == "mo-reacher-v4" or self.env_name == "mo-hopper-v4":
                    obs, reward, done, extra, info = self.env.step(action)
                    done = (done or extra)
                    #for i in range(len(reward)):
                    #    true_obj_episode_reward[i] += reward[i]
                    reward = sum(reward)
                else:
                    obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.disagreement_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                if self.multi_obj:
                    total_acc = np.mean(train_acc)
                else:
                    total_acc = train_acc


    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        episode_custom_reward = 0
        if self.env_name == "mo-halfcheetah-v4":
            true_obj_episode_reward = [0, 0, 0]
        elif self.env_name == "mo-reacher-v4":
            true_obj_episode_reward = [0, 0, 0, 0]
        elif self.env_name == "mo-hopper-v4":
            true_obj_episode_reward = [0, 0, 0, 0]
        else:
            true_obj_episode_reward = [0, 0, 0]
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 1000:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 1000 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/custom_reward', episode_custom_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                """WANDB LOGGING UNCOMMENT IF YOU WANT TO UTILIZE W&B"""
                """
                if wandb.run is not None:
                    if self.env_name == "mo-halfcheetah-v4" or self.env_name == "walker_walk":
                        wandb.log({'reward/episode_reward': episode_reward,
                                   'reward/true_episode_reward': true_episode_reward,
                                   'reward/true_obj_episode_reward1': true_obj_episode_reward[0],
                                   'reward/true_obj_episode_reward2': true_obj_episode_reward[1],
                                   'reward/custom_reward': episode_custom_reward,
                                   'reward/total_feedback': self.total_feedback,
                                   'reward/labeled_feedback': self.labeled_feedback,
                                   'reward/step': self.step})
                    elif self.env_name == "mo-reacher-v4":
                        wandb.log({'reward/episode_reward': episode_reward,
                                   'reward/true_episode_reward': true_episode_reward,
                                   'reward/true_obj_episode_reward1': true_obj_episode_reward[0],
                                   'reward/true_obj_episode_reward2': true_obj_episode_reward[1],
                                   'reward/true_obj_episode_reward3': true_obj_episode_reward[2],
                                   'reward/true_obj_episode_reward4': true_obj_episode_reward[3],
                                   'reward/custom_reward': episode_custom_reward,
                                   'reward/total_feedback': self.total_feedback,
                                   'reward/labeled_feedback': self.labeled_feedback,
                                   'reward/step': self.step})
                    elif self.env_name == "mo-hopper-v4":
                        wandb.log({'reward/episode_reward': episode_reward,
                                   'reward/true_episode_reward': true_episode_reward,
                                   'reward/true_obj_episode_reward1': true_obj_episode_reward[0],
                                   'reward/true_obj_episode_reward2': true_obj_episode_reward[1],
                                   'reward/true_obj_episode_reward3': true_obj_episode_reward[2],
                                   'reward/custom_reward': episode_custom_reward,
                                   'reward/total_feedback': self.total_feedback,
                                   'reward/labeled_feedback': self.labeled_feedback,
                                   'reward/step': self.step})
                    else:
                        wandb.log({'reward/episode_reward': episode_reward,
                                   'reward/true_episode_reward': true_episode_reward,
                                   'reward/true_obj_episode_reward1': true_obj_episode_reward[0],
                                   'reward/true_obj_episode_reward2': true_obj_episode_reward[1],
                                   'reward/true_obj_episode_reward3': true_obj_episode_reward[2],
                                   'reward/custom_reward': episode_custom_reward,
                                   'reward/total_feedback': self.total_feedback,
                                   'reward/labeled_feedback': self.labeled_feedback,
                                   'reward/step': self.step})
                """
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)

                    """WANDB LOGGING UNCOMMENT IF YOU WANT TO UTILIZE W&B"""
                    """
                    if wandb.run is not None:
                        wandb.log({'reward/episode_success': episode_success,
                                   'reward/true_episode_success': episode_success})
                    """
                if self.env_name == "mo-halfcheetah-v4" or self.env_name == "mo-reacher-v4" or self.env_name == "mo-hopper-v4":
                    obs, info = self.env.reset()
                else:
                    obs = self.env.reset()

                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.env_name == "mo-halfcheetah-v4":
                    true_obj_episode_reward = [0, 0, 0]
                elif self.env_name == "mo-reacher-v4":
                    true_obj_episode_reward = [0, 0, 0, 0]
                elif self.env_name == "mo-hopper-v4":
                    true_obj_episode_reward = [0, 0, 0, 0]
                else:
                    true_obj_episode_reward = [0, 0, 0]
                episode_custom_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)

            #next_obs, reward, done, extra = self.env.step(action)
            obj_rewards = []
            if self.env_name == "mo-halfcheetah-v4" or self.env_name == "mo-reacher-v4" or self.env_name == "mo-hopper-v4":
                next_obs, reward, done, extra, info = self.env.step(action)
                done = (done or extra)
            else:
                next_obs, reward, done, extra = self.env.step(action)
                reward = self.get_custom_reward_mo(next_obs, reward)

            for i in range(len(reward)):
                true_obj_episode_reward[i] += reward[i]
                obj_rewards.append(reward[i])
            reward = sum(reward)
            obj_rewards.insert(0, reward)

            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, obj_rewards, (self.step + 1) % 1000 == 0)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        for filename in os.listdir(self.work_dir):
            print(filename)
            f = os.path.join(self.work_dir, filename)
            wandb.save(f)

    def get_custom_reward(self, observation):
        if self.env_name == "walker_walk":  # Add reward based on z-axis
            return observation[0]
        if self.env_name == "cheetah_run":
            return observation[0]
        if self.env_name == "quadruped_walk":
            return observation[0]
        else:
            return min(observation[0], 0)

    def get_custom_reward_mo(self, observation, reward):
        if self.env_name == "walker_walk":  # Add reward based on z-axis
            return [reward, observation[0]]
        if self.env_name == "cheetah_run":
            return observation[0]
        if self.env_name == "quadruped_walk":
            return observation[0]
        if self.env_name == "metaworld_drawer-close-v2" or self.env_name == "metaworld_button-press-v2" or self.env_name == "metaworld_button-press-wall-v2" or self.env_name == "metaworld_window-open-v2":
            hand_pos = [observation[0], observation[1], observation[2]]
            obj_pos = [observation[4], observation[5], observation[6]]
            distance = math.sqrt((obj_pos[0] - hand_pos[0]) ** 2 + (obj_pos[1] - hand_pos[1]) ** 2 + (obj_pos[2] - hand_pos[2]) ** 2)
            return [reward, -distance, -1 if observation[0] > 0 else 0]
        else:
            return [reward]

@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()