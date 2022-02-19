import io
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import datasets, transforms
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    #继承torch.utils.data.Dataset,自己重写其中的 __init__ , __len__ , __getitem__ 
    def __init__(self,root, datatxt, transform, target_transform): #初始化一些需要传入的参数
        fh = open(root + datatxt, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []                      #创建一个名为img的空列表，一会儿用来装东西
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()   #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                                        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
    def __getitem__(self, index):    #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img=cv2.imread(fn)
        #img2 = Image.open(fn).convert('RGB') #按照path读入图片from PIL import Image # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img,label  #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
   
def img_show(dataset):
    img,label=dataset[1]           #获取传回的tensor和int
    img=img*255                         #反Normalize
    img=np.array(img,dtype='uint8')     #转为array
    img=img.transpose(1,2,0)            #更换维度
    cv2.imshow(str(label),img)          #显示图片
    cv2.waitKey()

#根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
root='./'

datatxt='test_data/label.txt'
test_dataset=MyDataset(root=root,datatxt=datatxt, transform=transforms.ToTensor(),target_transform=None)

datatxt='train_data/label.txt'
train_dataset=MyDataset(root=root,datatxt=datatxt, transform=transforms.ToTensor(),target_transform=None)

