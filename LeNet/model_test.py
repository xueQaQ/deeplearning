import torch
import torch.utils.data as Data

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root="./data", 
                          train=False,
                          transform=transforms.Compose([transforms.Resize(size=224),    # 将28x28图像上采样到224x224
                                                        transforms.ToTensor()           # 转换为张量并归一化到[0,1]
                                                        ]),
                          download=True)
    
    
    # 创建数据加载器
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,       #shuffle 是否打乱顺序， True，打乱，False，不打乱
                                       num_workers=2)      #使用8个子进程加载数据


    return test_dataloader


def test_model_process(model,test_dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #将模型放入设备当中
    model = model.to(device)

    #初始化参数
    test_corrects = 0.0
    test_num = 0

    #将梯度置为0，只进行前向传播，节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader :
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            
            #设置为验证模式
            model.eval() 
            #前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            ouput = model(test_data_x)
            #查找最大索引下标
            pre_lab = torch.argmax(ouput,dim=1)
            #如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            #将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    #计算测试准确率
    test_acc = test_corrects.double().item()/test_num
    print("测试的准确率为：",test_acc)

    return test_acc



if __name__ == "__name__":
    #加载模型
    model = LeNet()

    #模型序列化
    model.load_state_dict(torch.load('best_model.pth'))
    
    #加载测试数据集
    test_dataloader = test_data_process()

    #加载模型测试的函数
    # test_data_process(model,test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #设置模型为验证模式
            model.eval()
            output = model(b_x)

            #查找最大索引下标
            pre_lab = torch.argmax(ouput,dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值:",classes[result],"------","真实值",classes[label])