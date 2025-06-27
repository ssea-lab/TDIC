import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

class TDIC(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight, tide_weight):
        super(TDIC, self).__init__()

        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.q = nn.Parameter(torch.ones(num_items) * 0.1)
        self.b = nn.Parameter(torch.ones(num_items) * 0.1)

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.tide_weight = tide_weight

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

    def forward(self, user, item_p, item_n, mask):

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

        pop_p = F.softplus(self.q[item_p]) + F.softplus(self.b[item_p])
        pop_n = F.softplus(self.q[item_n]) + F.softplus(self.b[item_n])

        p_score_tide = torch.tanh(pop_p) * p_score_total
        n_score_tide = torch.tanh(pop_n) * n_score_total

        loss_tide = self.bpr_loss(p_score_tide, n_score_tide)


        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + self.tide_weight * loss_tide

        return loss

    def mask_bpr_loss(self, p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        #item_embeddings = self.items_pop
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        #user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')

class LGNTDIC(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight, pop_weight, tide_weight):
        super(LGNTDIC, self).__init__()

        self.n_user = num_users
        self.n_item = num_items


        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.q = nn.Parameter(torch.ones(num_items) * 0.1)
        self.b = nn.Parameter(torch.ones(num_items) * 0.1)

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.tide_weight = tide_weight

        # LGN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()

        self.dis_pen = dis_pen
        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.embeddings_int.size(1))
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.q)
        nn.init.uniform_(self.b)

    def forward(self, user, item_p, item_n, mask, graph, training=True):
        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = features_int[user]
        users_pop = features_pop[user]
        items_p_int = features_int[item_p]
        items_p_pop = features_pop[item_p]
        items_n_int = features_int[item_n]
        items_n_pop = features_pop[item_n]

        p_score_int = torch.sum(users_int * items_p_int, 2)
        n_score_int = torch.sum(users_int * items_n_int, 2)
        p_score_pop = torch.sum(users_pop * items_p_pop, 2)
        n_score_pop = torch.sum(users_pop * items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) - self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)

        pop_p = F.softplus(self.q[item_p - self.n_user]) + F.softplus(self.b[item_p - self.n_user])
        pop_n = F.softplus(self.q[item_n - self.n_user]) + F.softplus(self.b[item_n - self.n_user])

        p_score_tide = torch.tanh(pop_p) * p_score_total
        n_score_tide = torch.tanh(pop_n) * n_score_total

        loss_tide = self.bpr_loss(p_score_tide, n_score_tide)


        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + self.tide_weight * loss_tide

        return loss
