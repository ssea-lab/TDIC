#!/usr/local/anaconda3/envs/torch-1.1-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import torch

import data_utils.sampler as SAMPLER


class DataSampler(SAMPLER.Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate, popularity, margin=10, pool=10):

        super(DataSampler, self).__init__(flags_obj, lil_record, dok_record, neg_sample_rate)
        self.popularity = popularity
        self.margin = margin
        self.pool = pool

    def adapt(self, epoch, decay):

        self.margin = self.margin*decay

    def generate_negative_samples(self, user, pos_item):

        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=bool)

        user_pos = self.lil_record.rows[user]

        item_pos_pop = self.popularity[pos_item]

        pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
        unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
        unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
        num_unpop_items = len(unpop_items)

        if num_pop_items < self.pool:
            
            for count in range(self.neg_sample_rate):

                index = np.random.randint(num_unpop_items)
                item = unpop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]

                negative_samples[count] = item
                mask_type[count] = False

        elif num_unpop_items < self.pool:
            
            for count in range(self.neg_sample_rate):

                index = np.random.randint(num_pop_items)
                item = pop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]

                negative_samples[count] = item
                mask_type[count] = True
        
        else:

            for count in range(self.neg_sample_rate):

                if np.random.random() < 0.5:

                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_pop_items)
                        item = pop_items[index]

                    negative_samples[count] = item
                    mask_type[count] = True

                else:

                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_unpop_items)
                        item = unpop_items[index]

                    negative_samples[count] = item
                    mask_type[count] = False

        return negative_samples, mask_type

    def sample(self, index):

        user, pos_item = self.get_pos_user_item(index)

        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg, mask_type = self.generate_negative_samples(user, pos_item=pos_item)

        return users, items_pos, items_neg, mask_type
