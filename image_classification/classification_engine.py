import torch
from tqdm import tqdm


# 训练一个轮次
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    # 训练
    model.train()
    train_loss = 0
    total_num=0
    for input,target in train_loader:
        input=input.to(device)
        target=target.to(device)
        #核心步骤
        output=model(input)
        loss=loss_fn(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #累加损失
        train_loss += loss.item()*input.size(0)
        total_num+=input.size(0)
        #计算本轮损失
    this_loss=train_loss/total_num

    # 本轮训练结束，返回平均损失
    return this_loss

# 验证
def validate(model, val_loader, loss_fn, device):
    # 验证
    model.eval()
    total_loss = 0
    total_num=0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)

            total_loss += loss.item()*input.size(0)
            total_num+=input.size(0)
    # 返回平均损失
    return total_loss / total_num

# 测试
def evaluate(model, test_loader,  device):
    # 测试
    model.eval()
    total_num=0
    test_acc_num=0
    with torch.no_grad():
        for input, target in tqdm(test_loader, desc='测试'):
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            #得到预测分类标签
            perd=output.argmax(dim=-1)
            test_acc_num+=perd.eq(target).sum().item()
            total_num+=input.size(0)
    # 返回准确率
    return test_acc_num/ total_num
