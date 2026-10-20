import os
from math import ceil
import chromadb
import numpy as np
import torch
from PIL import Image
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from tqdm import tqdm
from common.utils import sorted_alphanum
from image_similarity.similarity_config import IMG_PATH, IMG_H, IMG_W, PACKAGE_NAME, CHROMA_BACKED_PATH, \
    IMAGE_COLLECTION_NAME, CHROMA_INSERT_BATCH, ENCODER_MODEL_NAME
import torchvision.transforms as T
from image_similarity.similarity_model import ConvEncoder


# 嵌入函数
class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    # 初始化，传入自己的嵌入模型（encoder）
    def __init__(self, model) -> None:
        self.model = model.to('cpu')

    # 调用方法
    def __call__(self, input: Images) -> Embeddings:
        # 将输入图像转换为 Tensor
        input_tensor = torch.tensor(np.array(input))
        # 前向传播，得到模型的输出
        with torch.no_grad():
            output = self.model(input_tensor)
        # 转成ndarray返回
        return output.numpy()

#加载所有图片，返回字典{ID，image}
def get_id2images(image_dir,transform):
    id2images = {}
    image_names=sorted_alphanum(os.listdir(image_dir))
    with tqdm(total=len(image_names)) as pbar:
        for i,image_name in enumerate(image_names):
            # 1. 构建图片的完整访问路径
            image_path = os.path.join(image_dir, image_names[i])
            # 2. 打开图片
            image = Image.open(image_path).convert('RGB')
            # 3. 应用转换操作，得到Tensor
            img_tensor = transform(image)
            #4.转换成ndarray，保存到字典
            id2images[str(i)] = img_tensor.numpy()
            pbar.update(1)
    return id2images



#获取chroma集合
def get_embedding_collection(encoder):
    path=os.path.join('..',PACKAGE_NAME, CHROMA_BACKED_PATH)
    client= chromadb.PersistentClient(path)
    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        embedding_function=ImageEmbeddingFunction(encoder)
    )
    return collection



#生成所有图像的嵌入向量（预处理）
def cerate_embeddings(encoder):
    # 定义图像转换操作（调整大小，转为Tensor）
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    #1加载所有图片，返回字典{ID，image}
    print("正在加载所有图片...")
    id2images = get_id2images(IMG_PATH,transform)
    print("图片加载完毕!!!")
    ids=list(id2images.keys())
    imgs=list(id2images.values())
    #2获取chroma集合
    collection = get_embedding_collection(encoder)

    #执行写入chroma操作
    print("正在向chorma写入数据...")
    #分批写入
    batchs=ceil(len(ids)/CHROMA_INSERT_BATCH)
    for i in tqdm(range(batchs),desc="写入数据"):
        start = i*CHROMA_INSERT_BATCH
        end = min((i+1)*CHROMA_INSERT_BATCH, len(ids))
        collection.upsert(
            ids=ids[start:end],
            images=imgs[start:end],
        )
    print("写入数据完成")



#相似图片搜索
def search_similar_image_ids(collection,image,cnt):
    result = collection.query(
        query_images=image.numpy(),
        n_results=cnt,
    )
    similar_image_ids = [int(id) for id in result["ids"][0]]
    return similar_image_ids


if __name__ == "__main__":
    encoder=ConvEncoder()
    state_dict=torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(state_dict)
    print("模型加载完毕！")
    cerate_embeddings(encoder)
