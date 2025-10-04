#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import dgl.function as fn

import utils

from deprecated import deprecated
from tqdm import tqdm

import random


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]

        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):

        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')



class TDIC(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight, tdic_weight, 
                 timestamp_popularity_calculator=None, use_timestamp_popularity=False):
        super(TDIC, self).__init__()

        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.q = nn.Parameter(torch.ones(num_items) * 0.1)
        self.b = nn.Parameter(torch.ones(num_items) * 0.1)

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.tdic_weight = tdic_weight
        
        self.timestamp_popularity_calculator = timestamp_popularity_calculator
        self.use_timestamp_popularity = use_timestamp_popularity

        # loss
        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()

        self.dis_pen = dis_pen
        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.users_int.size(1))
        self.users_int.data.uniform_(-stdv, stdv)
        self.users_pop.data.uniform_(-stdv, stdv)
        self.items_int.data.uniform_(-stdv, stdv)
        self.items_pop.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.q)
        nn.init.uniform_(self.b)

    def forward(self, user, item_p, item_n, mask, timestamp=None):

        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        items_p_int = self.items_int[item_p]
        items_p_pop = self.items_pop[item_p]
        items_n_int = self.items_int[item_n]
        items_n_pop = self.items_pop[item_n]

        p_score_int = torch.sum(users_int * items_p_int, 2)
        n_score_int = torch.sum(users_int * items_n_int, 2)
        p_score_pop = torch.sum(users_pop * items_p_pop, 2)
        n_score_pop = torch.sum(users_pop * items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)


        if self.use_timestamp_popularity and self.timestamp_popularity_calculator is not None and timestamp is not None:
            pop_p = self._calculate_timestamp_popularity(item_p, timestamp)
            pop_n = self._calculate_timestamp_popularity(item_n, timestamp)
        else:
            pop_p = F.softplus(self.q[item_p]) + F.softplus(self.b[item_p])
            pop_n = F.softplus(self.q[item_n]) + F.softplus(self.b[item_n])

        p_score_tdic = torch.tanh(pop_p) * p_score_total
        n_score_tdic = torch.tanh(pop_n) * n_score_total

        loss_tdic = self.bpr_loss(p_score_tdic, n_score_tdic)

        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + self.tdic_weight * loss_tdic

        return loss

    def mask_bpr_loss(self, p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))
    
    def _calculate_timestamp_popularity(self, item_ids, timestamp):

        item_ids_np = item_ids.cpu().numpy()
        timestamp_np = timestamp.cpu().numpy()
        
        popularities = []
        for i in range(len(item_ids_np)):
            item_id = item_ids_np[i]
            current_time = timestamp_np[i]
            popularity = self.timestamp_popularity_calculator.calculate_power_law_popularity_numpy(
                item_id, current_time
            )
            popularities.append(popularity)
        
        popularity_tensor = torch.from_numpy(np.array(popularities)).float().to(item_ids.device)
        
       
        q_values = F.softplus(self.q[item_ids])
        b_values = F.softplus(self.b[item_ids])
        
        combined_popularity = q_values + b_values * popularity_tensor
        
        return combined_popularity

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        #item_embeddings = self.items_pop
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        #user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')

