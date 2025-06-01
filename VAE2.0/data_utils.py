import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim

class MNISTDataset(Dataset):
    def __init__(self, dataset, shuffle=False):
        """
        Args:
            dataset: `datasets.Dataset`, including 'image' and 'label' keys。
            shuffle: Whether to shuffle as initialization。
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # transform the image into torch.tensor [0, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.dataset[real_idx]
        
        image = self.transform(sample['image'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, label

    def shuffle(self):
        random.shuffle(self.indices)

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def cal_fid(gen_images_with_labels, anc_imges_with_labels):
    fid = 0.
    for label in range(10):
        gen_images = np.array(gen_images_with_labels[label]).reshape(-1, 28 * 28) / 255
        anc_images = np.array(anc_imges_with_labels[label]).reshape(-1, 28 * 28) / 255
        gen_mean = np.mean(gen_images, axis=0)
        gen_var = np.var(gen_images, axis=0)
        anc_mean = np.mean(anc_images, axis=0)
        anc_var = np.var(anc_images, axis=0)
        fid += np.linalg.norm(gen_mean - anc_mean, ord=2) ** 2 + np.sum(gen_var + anc_var - 2 * (gen_var * anc_var) ** 0.5)
    
    return fid / 10.






