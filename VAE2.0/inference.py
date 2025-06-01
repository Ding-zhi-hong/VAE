import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.distributions import Normal, Bernoulli
import torch.nn.functional as F
import argparse
import os
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from tqdm import tqdm
from submission import VAE, GenModel
from data_utils import MNISTDataset, set_random_seed
from datasets import Dataset
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Inference model")

    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--task', type=str, default="VAEwolabel", help="Which task you are training, VAEwolabel or your Genwithlabel?")
    parser.add_argument('--checkpoint_path', type=str, default="VAEwolabel/valid/epoch_20.pth", help="Path to a checkpoint to load.")
    parser.add_argument('--latent_dim', type=int, default=20, help="The latent code's dimension.")
    parser.add_argument('--save_dir', type=str, default="VAEwolabel/test", help="Path to the inference results.")
    parser.add_argument('--data_path', type=str, default="dataset/valid/data.arrow", help="Path to the dataset for inference.")
    parser.add_argument('--inf_num', type=int, default=0, help="The number of data we will inference. If 0, means all of the test dataset will be used.")
    parser.add_argument('--batch_size', type=int, default=4096, help="batch size")

    return parser.parse_args()

def save_reconstructed(recon_x, save_dir, batch_idx, batch_size):
    recon_images = recon_x.cpu().view(-1, 1, 28, 28)
    for i, image in enumerate(recon_images):
        vutils.save_image(image, os.path.join(save_dir, f"recon_{batch_idx*batch_size+i}.png"), nrow=4)

def generate_images(model, inf_num, latent_dim, save_dir, device):
    os.makedirs(os.path.join(save_dir, 'gen'), exist_ok=True)
    for label in range(10):
        label_dir = os.path.join(save_dir, 'gen', str(label))
        os.makedirs(label_dir, exist_ok=True)
        for i in range(inf_num // 10):
            z = torch.randn(1, latent_dim).to(device)
            label = torch.tensor(label).to(device)
            z = z.to(device)
            with torch.no_grad():
                generated_image = model.decode(z, label).cpu().view(28, 28)
            generated_image = (generated_image.numpy() * 255).astype(np.uint8)
            generated_image = Image.fromarray(generated_image, mode='L')
            generated_image.save(os.path.join(label_dir, f"{i}.png"))

def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.task == "VAEwolabel":
        model = VAE(latent_dim=args.latent_dim)
    elif args.task == "Genwithlabel":
        model = GenModel(latent_dim=args.latent_dim)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = Dataset.from_file(args.data_path)
    inf_dataset = MNISTDataset(dataset, shuffle=False)
    inf_dataloader = DataLoader(inf_dataset, batch_size=args.batch_size, shuffle=False)
    inf_num = args.inf_num if args.inf_num else len(inf_dataset)

    print("********** Start Inference **********")
    print(f"Total inf num: {inf_num}, save_dir: {args.save_dir}, total batch: {(inf_num - 1) // args.batch_size + 1}")
    print("********** Reconstruction Task **********")
    os.makedirs(os.path.join(args.save_dir, 'recon'), exist_ok=True)

    for batch_idx, (images, labels) in tqdm(enumerate(inf_dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            recon_x, _, _ = model(images, labels)
        save_reconstructed(recon_x, os.path.join(args.save_dir, 'recon'), batch_idx, batch_size=args.batch_size)
        if (batch_idx + 1) * args.batch_size >= inf_num:
            break

    print("********** Generation Task **********")
    if args.task == "VAEwolabel":
        generate_images(model, 100, args.latent_dim, args.save_dir, device)
    elif args.task == "Genwithlabel":
        generate_images(model, 1000, args.latent_dim, args.save_dir, device)
    print("********** Inference Complete **********")

    # print(f"Negative log liklihood is : {nll}")

if __name__ == "__main__":
    main()
