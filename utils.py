import os
import datetime
import setproctitle

from absl import logging
from absl import flags
from visdom import Visdom

from deprecated import deprecated
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

import numbers

import torch

import data_utils.const as const_util
import trainer
# import recommender
# import candidate_generator as cg

import data_utils.loader as LOADER
import data_utils.transformer as TRANSFORMER
import data_utils.sampler as SAMPLER



class DatasetManager(object):

    def __init__(self, flags_obj):

        self.make_coo_loader_transformer(flags_obj)
        self.make_npy_loader(flags_obj)
        self.make_csv_loader(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):

        self.coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_transformer = TRANSFORMER.SparseTransformer(flags_obj)

    def make_npy_loader(self, flags_obj):

        self.npy_loader = LOADER.NpyLoader(flags_obj)

    def make_csv_loader(self, flags_obj):

        self.csv_loader = LOADER.CsvLoader(flags_obj)

    def get_dataset_info(self):

        coo_record = self.coo_loader.load(const_util.train_coo_record)

        self.num_users = coo_record.shape[0]
        self.num_items = coo_record.shape[1]

        self.coo_record = coo_record

    def get_skew_dataset(self):

        self.skew_coo_record = self.coo_loader.load(const_util.train_skew_coo_record)

    def get_popularity(self):

        self.popularity = self.npy_loader.load(const_util.popularity)
        return self.popularity

    def get_timestamps(self):
        self.timestamps = self.npy_loader.load(const_util.timestamps)
        return self.timestamps

    def get_blend_popularity(self):

        self.blend_popularity = self.npy_loader.load(const_util.blend_popularity)
        return self.blend_popularity



class TDICSampler(SAMPLER.Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate, popularity,  margin=10, pool=10):
        super(TDICSampler, self).__init__(flags_obj, lil_record, dok_record, neg_sample_rate)
        self.popularity = popularity
        self.margin = margin
        self.pool = pool

    def generate_negative_samples(self, user, pos_item):
        """
        依据用户时间戳生成负样本
        """
        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=bool)

        user_pos = self.lil_record.rows[user]
        item_pos_pop = self.popularity[pos_item]



        pop_items = np.nonzero(self.popularity > item_pos_pop +self.margin)[0]
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        unpop_items = np.nonzero(self.popularity < item_pos_pop / 2)[0]
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