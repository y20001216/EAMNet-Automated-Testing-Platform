import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

def cv_random_flip(img, label, grad):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        grad = grad.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, grad


def randomCrop(image, label, grad):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), grad.crop(random_region)


def randomRotation(image, label, grad):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        grad = grad.rotate(random_angle, mode)
    return image, label, grad


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, grad_root, trainsize):
        self.trainsize = trainsize
        image_list = [f for f in os.listdir(image_root) if f.endswith('.jpg')]
        gt_list = os.listdir(gt_root)
        grad_list = os.listdir(grad_root)

        self.images = []
        self.gts = []
        self.grads = []

        for img_name in image_list:
            base_name = os.path.splitext(img_name)[0]
            gt_name = base_name + '.png'
            grad_name = base_name + '.png'

            gt_path = os.path.join(gt_root, gt_name)
            grad_path = os.path.join(grad_root, grad_name)
            img_path = os.path.join(image_root, img_name)

            if os.path.exists(gt_path) and os.path.exists(grad_path):
                self.images.append(img_path)
                self.gts.append(gt_path)
                self.grads.append(grad_path)
            else:
                print(f'[WARNING] Missing GT or Grad for {base_name}')

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        # read assest/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        grad = self.binary_loader(self.grads[index])

        # data augumentation
        image, gt, grad = cv_random_flip(image, gt, grad)
        image, gt, grad = randomCrop(image, gt, grad)
        image, gt, grad = randomRotation(image, gt, grad)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        grad = self.gt_transform(grad)

        return image, gt, grad

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, grad_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, grad_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize, grad_root=None):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.grads = []
        if grad_root is not None: # 只有当提供了 grad_root 时，才加载 grads
            self.grads = [os.path.join(grad_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
            self.grads = sorted(self.grads)
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.grads = sorted(self.grads)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def load_grad(self, name):
        if not self.grads: # 如果 grads 列表为空
            print("[Warning] grad_root not provided, cannot load GT edge.")
            return np.zeros((self.testsize, self.testsize), dtype=np.uint8) # 返回一个黑色图像作为占位符
            
        current_index = (self.index - 1 + self.size) % self.size
        grad = self.binary_loader(self.grads[current_index])
        return np.array(grad)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
