import os

from PIL import Image

from common.utils import seed_everything
from image_similarity.similarity_data import create_datasets
from image_similarity.similarity_embeddings import search_similar_image_ids, get_embedding_collection
from image_similarity.similarity_model import ConvEncoder
from similarity_config import *
import torch
import matplotlib.pyplot as plt

if __name__=="__main__":
    seed_everything(SEED)

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建数据集
    _, _, test_dataset = create_datasets()
    print("===========数据集创建完成===========")

    # 3. 从测试集获取图片
    image,_=test_dataset[0]
    print(image.shape)


    # 4. 加载模型
    model = ConvEncoder().to(device)
    state_dict = torch.load(ENCODER_MODEL_NAME)
    model.load_state_dict( state_dict )
    print("===========模型加载完成===========")
    # 5. 获取chroma集合
    collection=get_embedding_collection(model)

    # 5. 测试
    similarity_image_ids=search_similar_image_ids(collection,image=image,cnt=5)
    print("相似图片id：",similarity_image_ids)

    #画图显示
    fig,axes=plt.subplots(2,5,figsize=(20,20))
    #输入图片
    image=image.permute(1,2,0).cpu().numpy()
    axes[0,0].imshow(image)
    for i in range(len(similarity_image_ids)):
        image_name=str(similarity_image_ids[i])+".jpg"
        image=Image.open(os.path.join(IMG_PATH,image_name))
        axes[1,i].imshow(image)
    for ax in axes.flatten():
        ax.axis('off')
    plt.show()
