from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import utils
import data_utils.loader as LOADER
import data_utils.transformer as TRANSFORMER
import data_utils.sampler as SAMPLER
import data_utils.const as const_util
import numpy as np
import scipy.sparse as sp


def get_dataloaders(flags_obj, dm):
    train_dataset = TDICFactorizationDataset(flags_obj, dm)
    train_dataloader = DataLoader(train_dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle,
                                  num_workers=flags_obj.num_workers, drop_last=True)

    val_dataset = TDICFactorizationDataset(flags_obj, dm)
    val_dataloader = DataLoader(val_dataset, batch_size=flags_obj.batch_size, shuffle=False,
                                num_workers=flags_obj.num_workers, drop_last=True)

    return train_dataloader


class FactorizationDataProcessor(object):

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_fdp'

    @staticmethod
    def get_TDIC_dataloader(flags_obj, dm):
        dataset = TDICFactorizationDataset(flags_obj, dm)
        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle,
                          num_workers=flags_obj.num_workers, drop_last=True)



class FactorizationDataset(Dataset):

    def __init__(self, flags_obj, dm):

        self.make_sampler(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):

        train_coo_record = dm.coo_record

        transformer = TRANSFORMER.SparseTransformer(flags_obj)
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        self.make_sampler_core(flags_obj, train_lil_record, train_dok_record)

    def __len__(self):

        return len(self.sampler.record)

    def __getitem__(self, index):

        raise NotImplementedError
class TDICFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):
        super(TDICFactorizationDataset, self).__init__(flags_obj, dm)
        self.make_sampler(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):
        dm.get_popularity()
        # dm.get_timestamps()

        transformer = TRANSFORMER.SparseTransformer(flags_obj)
        train_coo_record = dm.coo_record
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        self.sampler = utils.TDICSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate, dm.popularity, margin=flags_obj.margin, pool=flags_obj.pool)

        train_skew_coo_record = dm.skew_coo_record
        train_skew_lil_record = transformer.coo2lil(train_skew_coo_record)
        train_skew_dok_record = transformer.coo2dok(train_skew_coo_record)

        self.skew_sampler = utils.TDICSampler(flags_obj, train_skew_lil_record, train_skew_dok_record, flags_obj.neg_sample_rate, dm.popularity,  margin=flags_obj.margin, pool=flags_obj.pool)

    def __len__(self):
        return len(self.sampler.record) + len(self.skew_sampler.record)

    def __getitem__(self, index):
        # 根据索引采样，返回带有时间戳的样本
        if index < len(self.sampler.record):
            users, items_p, items_n, mask= self.sampler.sample(index)
            mask = torch.BoolTensor(mask)
        else:
            users, items_p, items_n, mask = self.skew_sampler.sample(index - len(self.sampler.record))
            mask = torch.BoolTensor(mask)

        return users, items_p, items_n, mask

    def adapt(self, epoch, decay):
        self.sampler.adapt(epoch, decay)
        self.skew_sampler.adapt(epoch, decay)



class CGDataProcessor(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.model_type

    @staticmethod
    def get_dataloader(flags_obj, test_data_source):

        dataset = CGDataset(flags_obj, test_data_source)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=True, num_workers=flags_obj.num_workers, drop_last=False), dataset.max_train_interaction


class CGDataset(Dataset):

    def __init__(self, flags_obj, test_data_source):

        self.test_data_source = test_data_source
        self.sort_users(flags_obj)

    def sort_users(self, flags_obj):

        loader = LOADER.CooLoader(flags_obj)
        if self.test_data_source == 'val':
            coo_record = loader.load(const_util.val_coo_record)
        elif self.test_data_source == 'test':
            coo_record = loader.load(const_util.test_coo_record)
        transformer = TRANSFORMER.SparseTransformer(flags_obj)
        self.lil_record = transformer.coo2lil(coo_record)

        train_coo_record = loader.load(const_util.train_coo_record)
        train_skew_coo_record = loader.load(const_util.train_skew_coo_record)

        blend_user = np.hstack((train_coo_record.row, train_skew_coo_record.row))
        blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))
        blend_value = np.hstack((train_coo_record.data, train_skew_coo_record.data))
        blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)), shape=train_coo_record.shape)

        self.train_lil_record = transformer.coo2lil(blend_coo_record)

        train_interaction_count = np.array([len(row) for row in self.train_lil_record.rows], dtype=np.int64)
        self.max_train_interaction = int(max(train_interaction_count))

        test_interaction_count = np.array([len(row) for row in self.lil_record.rows], dtype=np.int64)
        self.max_test_interaction = int(max(test_interaction_count))

    def __len__(self):

        return len(self.lil_record.rows)

    def __getitem__(self, index):

        unify_train_pos = np.full(self.max_train_interaction, -1, dtype=np.int64)
        unify_test_pos = np.full(self.max_test_interaction, -1, dtype=np.int64)

        train_pos = self.train_lil_record.rows[index]
        test_pos = self.lil_record.rows[index]

        unify_train_pos[:len(train_pos)] = train_pos
        unify_test_pos[:len(test_pos)] = test_pos

        return torch.LongTensor([index]), torch.LongTensor(unify_train_pos), torch.LongTensor(unify_test_pos), torch.LongTensor([len(test_pos)])