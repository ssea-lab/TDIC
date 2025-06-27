# tester.py

import torch
import numpy as np
from tqdm import tqdm
from absl import logging
 # 导入 config 文件
from metrics import Judger  # 假设你已经定义了 Judger 类
import faiss

class FaissInnerProductMaximumSearchGenerator:
    def __init__(self, flags_obj, items):
        self.items = items
        self.embedding_size = items.shape[1]
        self.make_index(flags_obj)

    def make_index(self, flags_obj):
        self.make_index_brute_force(flags_obj)

        if flags_obj.cg_use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, flags_obj.cg_gpu_id, self.index)

    def make_index_brute_force(self, flags_obj):
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.items)

    def generate(self, users, k):
        _, I = self.index.search(users, k)
        return I

    def generate_with_distance(self, users, k):
        D, I = self.index.search(users, k)
        return D, I

class Tester:
    def __init__(self, config, model, dm,device):
        self.config = config
        self.model = model
        self.dm = dm
        self.judger = Judger(config, dm, max(config.topk))
        self.device = device
        self.max_topk = max(config.topk)
        self.results = {k: 0.0 for k in self.judger.metrics}

    def evaluate(self, dataloader,margin_topk):
        self.model.eval()
        total_loss = 0
        all_items = []
        all_test_pos = []
        all_num_test_pos = []
        self.make_cg()
        real_num_test_users = 0

        with torch.no_grad():
            for batch_count, data in enumerate(tqdm(dataloader)):

                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()

                items = self.cg(users, max(self.config.topk)+margin_topk)

                items = self.filter_history(items, train_pos)

                batch_results, valid_num_users = self.judger.judge(items, test_pos, num_test_pos)

                real_num_test_users = real_num_test_users + valid_num_users

                for metric, value in batch_results.items():
                    self.results[metric] = self.results[metric] + value

        for metric, value in self.results.items():
            if metric in ['recall', 'hit_ratio', 'ndcg']:
                self.results[metric] = value/real_num_test_users

        return self.results

    def make_cg(self):
        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = FaissInnerProductMaximumSearchGenerator(self.config, self.item_embeddings)
        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):
        return self.generator.generate(self.user_embeddings[users], topk)

    def filter_history(self, items, train_pos):

        return np.stack([items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))], axis=0)