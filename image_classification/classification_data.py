import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from classification_config import  *
from common.utils import sorted_alphanum

class ImageLabelDataset(Dataset):
    # 初始化
    def __init__(self, image_dir,label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))
        label_data = pd.read_csv(label_path)
        self.lebels=label_data.iloc[:,1].values
    # 获取数据集长度
    def __len__(self):
        return len(self.image_names)
    # 根据索引号获取元素：(input, target) = (noise_image, image)
    def __getitem__(self, idx):
        # 1. 构建图片的完整访问路径
        import os
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # 2. 打开图片
        image = Image.open(image_path).convert('RGB')
        # 3. 应用转换操作，得到Tensor
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None！")
        # 4.找到图片对应的的分类标签
        image_label = int(self.lebels[idx])
        return  img_tensor,image_label


# 创建数据集并划分
def create_datasets():
    # 定义图像转换操作（调整大小，转为Tensor）
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    # 创建数据集
    dataset = ImageLabelDataset( IMG_PATH,LABLES_PATH ,transform=transform )

    #划分数据集
    generator = torch.Generator().manual_seed(SEED)
    train_dataset,test_dataset,val_dataset=random_split(dataset,[TRAIN_RATIO,TEST_RATIO,VAL_RATIO],generator=generator)
    return train_dataset,test_dataset,val_dataset

if __name__ == "__main__":
    train_dataset,test_dataset,val_dataset=create_datasets()
    print(f'traindataset {len(train_dataset)},testdataset {len(test_dataset)},valdataset {len(val_dataset)}')

