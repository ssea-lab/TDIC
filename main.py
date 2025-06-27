import torch
from trainer import Trainer
from dataloader import get_dataloaders
from config import Config
import utils


def main():
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = utils.DatasetManager(config)

    # 加载数据
    data.get_dataset_info()
    data.get_skew_dataset()
    train_dataloader= get_dataloaders(config, data)
    # 创建训练器
    trainer = Trainer(config, device,data)

    # 训练模型
    trainer.train(train_dataloader)

    trainer.test()

if __name__ == "__main__":
    main()