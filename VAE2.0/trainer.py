import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from submission import vae_loss
from data_utils import MNISTDataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, pred_step=100, max_checkpoints=5, eval_num=5000, 
                 batch_size=128, lr=1e-3, epochs=20, latent_dim=20, save_dir=None, load_from_checkpoint=False):
        """
        Args:
            - model: VAE model or your personal designed model
            - pred_step: During training, after pred_step, we will use validation to record the validation loss
            - dataset: A dataset object containing 'image' and 'label' fields.
            - batch_size: The number of samples per batch.
            - lr: Learning rate for the optimizer.
            - epochs: Number of training epochs.
            - save_dir: Directory to save model checkpoints.
            - load_from_checkpoint: Whether to load from latest checkpoint to continue training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.pred_step = pred_step
        self.train_dataset = MNISTDataset(train_dataset, shuffle=True)
        self.valid_dataset = MNISTDataset(valid_dataset, shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        self.eval_num = min(eval_num, len(self.valid_dataset))
        
        self.model = model.to(self.device)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.saved_checkpoints = []
        self.max_checkpoints = max_checkpoints
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"All trainable parameters: {trainable_params}")

        self.save_hyperparams(trainable_params)

        if load_from_checkpoint:
            self.load_latest_checkpoint()

    def train(self, show_loss_plot=False, save_loss_plot=True, var=0.5):
        """ Training loop for the VAE model. """
        self.model.train()
        self.cur_step = 0
        train_loss_list = [0.]
        valid_loss_list = [0.]
        print("********** Start training! **********")
        print(f"training args: lr: {self.lr}, total epochs: {self.epochs}, total steps: {self.epochs * ((len(self.train_dataset) + 1) // self.batch_size)}, \
var: {var}, total training data: {len(self.train_dataset)}, total validation data: {self.eval_num}, save path: {self.save_dir}")
        
        # Initialize tqdm progress bar
        pbar = tqdm(range(self.epochs), desc="Epoch", dynamic_ncols=True)
        
        for epoch in pbar:
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                self.cur_step += 1
                images = images.to(self.device).view(-1, 1, 28, 28)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                recon_x, mu, log_var = self.model(images, labels)
                loss = vae_loss(recon_x, images, mu, log_var, var=var)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() + images.size(0)
                train_loss_list.append(loss.item())
                
                if self.cur_step % self.pred_step == 0:
                    valid_loss_list.append(self.prediction_step(var).item())

                # Update the progress bar with current training loss
                pbar.set_postfix(train_loss=loss.item(), valid_loss=valid_loss_list[-1])

            avg_loss = total_loss / len(self.train_dataloader)
            
            # Save checkpoint
            self.save_checkpoint(epoch+1)
        
        print("********** Training complete! **********")
        self.plot_loss(train_loss_list[1:], valid_loss_list[1:], show=show_loss_plot, save=save_loss_plot)

        return train_loss_list[1:], valid_loss_list[1:]

    def save_checkpoint(self, epoch):
        """Saves the model checkpoint and deletes old checkpoints if necessary."""
        checkpoint_name = f"epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        if len(self.saved_checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        """ Loads a saved model checkpoint. """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def prediction_step(self, var):
        self.model.eval()
        total_loss = 0.
        cur_eval_num = 0
        for _, (images, labels) in enumerate(self.valid_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            recon_x, mu, log_var = self.model(images, labels)
            total_loss += vae_loss(recon_x, images, mu, log_var, var)
            cur_eval_num += 1
            if cur_eval_num * self.valid_dataloader.batch_size > self.eval_num:
                break

        self.model.train()
        return total_loss / cur_eval_num

    def save_hyperparams(self, params_num):
        """ Save training hyperparameters to a text file. """
        hyperparams = {
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "pred_step": self.pred_step,
            "save_dir": self.save_dir, 
            "latent_dim": self.latent_dim, 
            "trainable_parameters": params_num
        }
        with open(os.path.join(self.save_dir, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=4)

    def load_latest_checkpoint(self):
        """ Load the latest checkpoint in save_dir. """
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith(".pth")]
        if not checkpoints:
            print("No checkpoint found, starting training from scratch.")
            return
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.load_checkpoint(os.path.join(self.save_dir, latest_checkpoint))

    def plot_loss(self, train_loss, valid_loss, show=False, save=False):
        """ Plot training and validation loss. """
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss)), train_loss, label="Train Loss", alpha=0.7)
        if len(valid_loss) < len(train_loss):
            x_train = np.arange(len(train_loss))
            x_valid = np.linspace(0, len(train_loss) - 1, len(valid_loss))
            valid_loss_interpolated = np.interp(x_train, x_valid, valid_loss)
            plt.plot(x_train, valid_loss_interpolated, label="Valid Loss", alpha=0.7)
        else:
            plt.plot(range(0, len(train_loss), self.pred_step), valid_loss, label="Valid Loss", alpha=0.7)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
            print("Successfully saved the loss plot!")
        if show:
            plt.show()