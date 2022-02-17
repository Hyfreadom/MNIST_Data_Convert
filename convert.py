import os
import cv2
import numpy as np
import struct
def save_mnist_to_jpg(dataset_path, output_path): 
    files = os.listdir(dataset_path)
    print(files)
    #make image_file path
    mnist_image_file = os.path.join(dataset_path, [f for f in files if "image" in f][0])
    #make label_file path 
    mnist_label_file = os.path.join(dataset_path, [f for f in files if "label" in f][0])
    save_dir = output_path
    prefix = 'test'
    #read in binary
    with open(mnist_image_file, 'rb') as f1: 
        head_file=f1.read(16)# head_inf, read by the following function
        magic_number, num_file, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        size = rows*cols
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        magic_number,num_file = struct.unpack('>II',f2.read(8))       
        print(num_file)
        label_file = f2.read()
    for i in range(num_file):
        label = label_file[i]
        image_list = [item for item in image_file[i * size:i * size + size]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(height, width)
        save_name = os.path.join(save_dir, '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, image_np)
    print("=" * 20, "preprocess data finished", "=" * 20)
 
if __name__ == '__main__':
    save_mnist_to_jpg('./data/MNIST/test/','./data/MNIST/test_picture/')
    save_mnist_to_jpg('./data/MNIST/train/','./data/MNIST/train_picture/')
