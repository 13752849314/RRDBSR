import os
import csv
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ArchiveSet(Dataset):

    def __init__(self, path, model='train', width=300, height=200, grade='2', scala=4):
        super(ArchiveSet, self).__init__()
        self.path = path
        self.model = model
        self.width = width
        self.height = height
        self.folders = []
        self.grade = grade
        self.scala = scala
        for name in sorted(os.listdir(os.path.join(path)), reverse=True):
            if not os.path.isdir(os.path.join(path, name)):
                continue
            self.folders.append(name)
        self.lows, self.highs = self._load_csv('images.csv')

        if model == 'train':
            self.lows = self.lows[:int(0.6 * len(self.lows))]
            self.highs = self.highs[:int(0.6 * len(self.highs))]
        elif model == 'val':
            self.lows = self.lows[int(0.6 * len(self.lows)):int(0.8 * len(self.lows))]
            self.highs = self.highs[int(0.6 * len(self.highs)):int(0.8 * len(self.highs))]
        elif model == 'test':
            self.lows = self.lows[int(0.8 * len(self.lows)):]
            self.highs = self.highs[int(0.8 * len(self.highs)):]
        else:
            raise ValueError(f"model {model} is not available!")

    def __getitem__(self, index):
        low, high = self.lows[index], self.highs[index]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])
        tf1 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.height * self.scala, self.width * self.scala)),
            transforms.ToTensor()
        ])
        low = tf(low)
        high = tf1(high)
        name = self.lows[index].split('\\')[-1].split('.')[0]
        return {'LR': low, 'HR': high, 'name': name}

    def __len__(self):
        return len(self.lows)

    def _load_csv(self, filename):
        if not os.path.exists(os.path.join(self.path, filename)):
            images = []
            for i in os.listdir(os.path.join(self.path, self.folders[0])):
                num = int(str(i).split('_')[0])
                images.append((os.path.join(self.path, str(self.folders[0]), str(i)),
                               os.path.join(self.path, str(self.folders[-1]), str(num) + '.jpg')))

            print(len(images), images)

            random.shuffle(images)

            with open(os.path.join(self.path, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for i in images:
                    writer.writerow(i)

        low_images, high_images = [], []
        with open(os.path.join(self.path, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                low, high = row
                if low.split('_')[-1].split('.')[0] == self.grade:
                    low_images.append(low)
                    high_images.append(high)
                if self.grade == 'all':
                    low_images.append(low)
                    high_images.append(high)

        assert len(low_images) == len(high_images), '数据长度不一致!'

        return low_images, high_images


if __name__ == '__main__':
    s = ArchiveSet(r'../data/Archive')
    print(len(s.lows))
    print(s.folders)
    item = next(iter(s))
    print(item['name'])
