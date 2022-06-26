import imp
import pdb
from collections import OrderedDict
from copy import deepcopy
from distutils.command.config import config
from turtle import pd
from certifi import where

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections import ConfigDict
from torch import ne, nn as nn

from model import Scalar, soft_target_update
from utils_h2o import prefix_metrics


class Sim2realSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.batch_size = 256
        config.device = 'cuda'
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.d_sa_lr = 3e-4
        config.d_sas_lr = 3e-4
        config.d_early_stop_steps = 1000000
        config.noise_std_discriminator = 0.1
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = True
        config.use_variant = False
        config.u_ablation = False
        config.use_td_target_ratio = True
        config.use_sim_q_coeff = True
        config.use_kl_baseline = False
        config.fix_baseline_steps = 10
        # kl divergence: E_pM log(pM/pM^)
        config.sim_q_coeff_min = 1e-45
        config.sim_q_coeff_max = 10
        config.sampling_n_next_states = 10
        config.s_prime_std_ratio = 1.
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 0.01
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -1000
        config.cql_clip_diff_max = 1000
        # pM/pM^
        config.clip_dynamics_ratio_min = 1e-5
        config.clip_dynamics_ratio_max = 1
        config.sa_prob_clip = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, d_sa, d_sas, replay_buffer, dynamics_model=None):
        self.config = Sim2realSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.mean, self.std = self.replay_buffer.get_mean_std()
        self.next_observation_sampler = dynamics_model
        self.kl_baseline = 1

        '''
        Optimizers
        '''
        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )

        self.d_sa_optimizer = optimizer_class(self.d_sa.parameters(), self.config.d_sa_lr)
        self.d_sas_optimizer = optimizer_class(self.d_sas.parameters(), self.config.d_sas_lr)

        # whether to use automatic entropy tuning (True in default)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        # whether to use the lagrange version of CQL: False by default
        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, real_batch_size, sim_batch_size, bc=False):
        self._total_steps += 1

        #TODO Sim2real CQL
        real_batch = self.replay_buffer.sample(real_batch_size, scope="real")
        sim_batch = self.replay_buffer.sample(sim_batch_size, scope="sim")

        # real transitions from d^{\pi_\beta}_\mathcal{M}
        real_observations = real_batch['observations']
        real_actions = real_batch['actions']
        real_rewards = real_batch['rewards']
        real_next_observations = real_batch['next_observations']
        real_dones = real_batch['dones'] 

        # sim transitions from d^\pi_\mathcal{\widehat{M}}
        sim_observations = sim_batch['observations']
        sim_actions = sim_batch['actions']
        sim_rewards = sim_batch['rewards']
        sim_next_observations = sim_batch['next_observations']
        sim_dones = sim_batch['dones']
        
        # mixed transitions
        df_observations = torch.cat([real_observations, sim_observations], dim=0)
        df_actions =  torch.cat([real_actions, sim_actions], dim=0)
        df_rewards =  torch.cat([real_rewards, sim_rewards], dim=0)

        dsa_loss, dsas_loss = self.train_discriminator()

        # TODO new_action and log pi
        df_new_actions, df_log_pi = self.policy(df_observations)

        # True by default
        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (df_log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = df_observations.new_tensor(0.0)
            alpha = df_observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        # Improve policy under state marginal distribution d_f
        if bc:
            log_probs = self.policy.log_prob(df_observations, df_actions)
            policy_loss = (alpha * df_log_pi - log_probs).mean()
        else:
            # TODO
            q_new_actions = torch.min(
                self.qf1(df_observations, df_new_actions),
                self.qf2(df_observations, df_new_actions),
            )
            policy_loss = (alpha * df_log_pi - q_new_actions).mean() 

        """ Q function loss """
        # Q function in real data and sim data
        real_q1_pred = self.qf1(real_observations, real_actions)
        real_q2_pred = self.qf2(real_observations, real_actions)
        sim_q1_pred = self.qf1(sim_observations, sim_actions)
        sim_q2_pred = self.qf2(sim_observations, sim_actions)

        # False by default (enabling self.config.cql_max_target_backup)
        # TODO check if this is correct
        real_new_next_actions, real_next_log_pi = self.policy(real_next_observations)
        # TODO
        real_target_q_values = torch.min(
            self.target_qf1(real_next_observations, real_new_next_actions),
            self.target_qf2(real_next_observations, real_new_next_actions),
        )
        sim_new_next_actions, sim_next_log_pi = self.policy(sim_next_observations)
        # TODO
        sim_target_q_values = torch.min(
            self.target_qf1(sim_next_observations, sim_new_next_actions),
            self.target_qf2(sim_next_observations, sim_new_next_actions),
        )

        # False by default
        if self.config.backup_entropy:
            real_target_q_values = real_target_q_values - alpha * real_next_log_pi
            sim_target_q_values = sim_target_q_values - alpha * sim_next_log_pi
        real_td_target = torch.squeeze(real_rewards, -1) + (1. - torch.squeeze(real_dones, -1)) * self.config.discount * real_target_q_values
        sim_td_target = torch.squeeze(sim_rewards, -1) + (1. - torch.squeeze(sim_dones, -1)) * self.config.discount * sim_target_q_values

        real_qf1_loss = F.mse_loss(real_q1_pred, real_td_target.detach())
        real_qf2_loss = F.mse_loss(real_q2_pred, real_td_target.detach())

        # importance sampling on td error due to the dyanmics shift
        # TODO more elegant?
        if self.config.use_td_target_ratio:
            sqrt_IS_ratio = torch.clamp(self.real_sim_dynacmis_ratio(sim_observations, sim_actions, sim_next_observations), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max).sqrt()
        else:
            sqrt_IS_ratio = torch.ones((sim_observations.shape[0],)).to(self.config.device)
        sim_qf1_loss = F.mse_loss(sqrt_IS_ratio.squeeze() * sim_q1_pred, sqrt_IS_ratio.squeeze() * sim_td_target.detach())
        sim_qf2_loss = F.mse_loss(sqrt_IS_ratio.squeeze() * sim_q2_pred, sqrt_IS_ratio.squeeze() * sim_td_target.detach())

        qf1_loss = real_qf1_loss + sim_qf1_loss
        qf2_loss = real_qf2_loss + sim_qf2_loss

        ### Conservative Penalty loss: sim data
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            # shape [128]
            cql_q1 = self.qf1(sim_observations, sim_actions)
            cql_q2 = self.qf2(sim_observations, sim_actions)
            # TODO Q + log(u)
            if self.config.use_sim_q_coeff:
                u_sa = self.kl_sim_divergence(sim_observations, sim_actions, sim_next_observations)
            else:
                u_sa = torch.ones(sim_rewards.shape[0], device=self.config.device)
            
            omega = u_sa / u_sa.sum()
            std_omega  = omega.std()

            if self.config.u_ablation and self._total_steps % 1000 == 0:
                x_velocity  = sim_observations[:,8].cpu().detach().numpy().reshape((-1,1))
                u_log = u_sa.cpu().detach().numpy().reshape((-1,1))
                omega_log = omega.cpu().detach().numpy().reshape((-1,1))
                Q_log = sim_q1_pred.cpu().detach().numpy().reshape((-1,1))
                loggings = np.concatenate((x_velocity, u_log, omega_log, Q_log), axis=1)
                df = pd.DataFrame(loggings,columns=["velocity", "u_sa", "omega_sa", "Q value"])
                df.to_csv("ablation_log/step_{}-v3.csv".format(int(self._total_steps/1000)))

            # Q values on the actions sampled from the policy
            if self.config.use_variant:
                cql_qf1_gap = (omega * cql_q1).sum()
                cql_qf2_gap = (omega * cql_q2).sum()
            else:
                cql_q1 += torch.log(omega)
                cql_q2 += torch.log(omega)
                cql_qf1_gap = torch.logsumexp(cql_q1 / self.config.cql_temp, dim=0) * self.config.cql_temp
                cql_qf2_gap = torch.logsumexp(cql_q2 / self.config.cql_temp, dim=0) * self.config.cql_temp


            """Q values on the stat-actions with larger dynamics gap - Q values on data"""
            cql_qf1_diff = torch.clamp(
                cql_qf1_gap - real_q1_pred.mean(),
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            )
            cql_qf2_diff = torch.clamp(
                cql_qf2_gap - real_q2_pred.mean(),
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            )

            # False by default
            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = df_observations.new_tensor(0.0)
                alpha_prime = df_observations.new_tensor(0.0)

            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.total_steps % self.config.target_update_period == 0:
                self.update_target_network(
                    self.config.soft_target_update_rate
                )

            metrics = dict(
                dsa_train_loss=dsa_loss,
                dsas_train_loss=dsas_loss,
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_sim_qf1=sim_q1_pred.mean().item(),
                average_sim_qf2=sim_q2_pred.mean().item(),
                average_real_target_q=real_target_q_values.mean().item(),
                average_sim_target_q=sim_target_q_values.mean().item(),
                total_steps=self.total_steps,
            )

            if self.config.use_cql:
                metrics.update(prefix_metrics(dict(
                    u_sa=u_sa.mean().item(),
                    std_omega=std_omega.mean().item(),
                    sqrt_IS_ratio=sqrt_IS_ratio.mean().item(),
                    cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                    cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                    cql_qf1_diff=cql_qf1_diff.mean().item(),
                    cql_qf2_diff=cql_qf2_diff.mean().item(),
                    cql_qf1_gap=cql_qf1_gap.mean().item(),
                    cql_qf2_gap=cql_qf2_gap.mean().item(),
                ), 'cql'))

            return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    def train_discriminator(self):
        real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()

        # input noise: prevents overfitting
        if self.config.noise_std_discriminator > 0:
            real_obs += torch.randn(real_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            real_action += torch.randn(real_action.shape, device=self.config.device) * self.config.noise_std_discriminator
            real_next_obs += torch.randn(real_next_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_obs += torch.randn(sim_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_action += torch.randn(sim_action.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_next_obs += torch.randn(sim_next_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
        
        real_sa_logits = self.d_sa(real_obs, real_action)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(sim_obs, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)

        real_adv_logits = self.d_sas(real_obs, real_action, real_next_obs)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(sim_obs, sim_action, sim_next_obs)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)

        dsa_loss = (- torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()
        dsas_loss = (- torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()

        # Optimize discriminator(s,a) and discriminator(s,a,s')
        self.d_sa_optimizer.zero_grad()
        dsa_loss.backward(retain_graph=True)

        self.d_sas_optimizer.zero_grad()
        dsas_loss.backward()

        self.d_sa_optimizer.step()
        self.d_sas_optimizer.step()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()


    def discriminator_evaluate(self):
        s_real, a_real, next_s_real = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        s_sim, a_sim, next_s_sim = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()
        
        real_sa_logits = self.d_sa(s_real, a_real)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(s_sim, a_sim)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        dsa_loss = ( - torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()

        real_adv_logits = self.d_sas(s_real, a_real, next_s_real)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(s_sim, a_sim, next_s_sim)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)
        dsas_loss = ( - torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()


    def kl_sim_divergence(self, observations, actions, next_observations):
        # expectation on next observation over learned dynamics from the real offline data
        if self.next_observation_sampler == None:
            # TODO
            observations = torch.repeat_interleave(observations, self.config.sampling_n_next_states, dim=0)
            actions = torch.repeat_interleave(actions, self.config.sampling_n_next_states, dim=0)
            next_observations = torch.repeat_interleave(next_observations, self.config.sampling_n_next_states, dim=0)
            next_observations += torch.randn(next_observations.shape, device=self.config.device) * self.std * self.config.s_prime_std_ratio
            log_ratio = self.log_sim_real_dynacmis_ratio(observations, actions, next_observations).reshape((-1, self.config.sampling_n_next_states))

        else:
            n_next_observations = self.next_observation_sampler.get_next_state(observations, actions, self.config.sampling_n_next_states)
            sum_log_ratio = self.log_sim_real_dynacmis_ratio(observations, actions, n_next_observations[0])
            for i in range(1, self.config.sampling_n_next_states):
                sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, n_next_observations[i])

        # TODO
        return torch.clamp(log_ratio.mean(dim=1), self.config.sim_q_coeff_min, self.config.sim_q_coeff_max)


    def log_sim_real_dynacmis_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM^/pM
            log_ratio = - torch.log(sas_prob[:, 0]) \
                    + torch.log(sas_prob[:, 1]) \
                    + torch.log(sa_prob[:, 0]) \
                    - torch.log(sa_prob[:, 1])
        
        return log_ratio

    def log_real_sim_dynacmis_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            log_ratio = torch.log(sas_prob[:, 0]) \
                    - torch.log(sas_prob[:, 1]) \
                    - torch.log(sa_prob[:, 0]) \
                    + torch.log(sa_prob[:, 1])
        
        return log_ratio

    def real_sim_dynacmis_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            ratio = (sas_prob[:, 0] * sa_prob[:, 1]) / (sas_prob[:, 1] * sa_prob[:, 0])
        
        return ratio