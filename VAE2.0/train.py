import argparse
import os
from trainer import Trainer
from submission import VAE, GenModel
from datasets import Dataset
from data_utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model")

    # Required arguments
    parser.add_argument('--task', type=str, default="VAEwolabel", help="Which task you are training, VAEwolabel or your Genwithlabel?")
    
    parser.add_argument('--train_data', type=str, default="dataset/train/data.arrow", help="Path to the training dataset")
    parser.add_argument('--valid_data', type=str, default="dataset/valid/data.arrow", help="Path to the validation dataset")
    parser.add_argument('--save_dir', type=str, default="test", help="Directory to save checkpoints and logs")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to load (optional)")

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--latent_dim', type=int, default=20, help="Dimensionality of the latent space")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
    parser.add_argument('--pred_step', type=int, default=10, help="Steps between validation checks")
    parser.add_argument('--var', type=float, default=0.5, help="hyperparameter, constrains the weight of reconstruction loss.")

    # Optional arguments
    parser.add_argument('--eval_num', type=int, default=5000, help="Num of data used during validation step.")
    parser.add_argument('--load_from_checkpoint', action='store_true', help="Whether to load from the latest checkpoint")
    parser.add_argument('--show_loss_plot', type=bool, default=True, help="Whether to show the loss curve after training.")
    parser.add_argument('--save_loss_plot', type=bool, default=True, help="Whether to save the loss curve after training.")

    return parser.parse_args()

def main():
    args = parse_args()
    set_random_seed(args.seed)

    save_dir = os.path.join(args.task, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading training data from {args.train_data}")
    train_dataset = Dataset.from_file(args.train_data)
    print(f"Loading validation data from {args.valid_data}")
    valid_dataset = Dataset.from_file(args.valid_data)

    if args.task == "VAEwolabel":
        model = VAE(latent_dim=args.latent_dim)
    elif args.task == "Genwithlabel":
        model = GenModel(latent_dim=args.latent_dim)
    
    # Initialize the Trainer class
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=save_dir,
        load_from_checkpoint=args.load_from_checkpoint,
        pred_step=args.pred_step, 
        eval_num=args.eval_num
    )

    train_loss, valid_loss = trainer.train(show_loss_plot=args.show_loss_plot, save_loss_plot=args.save_loss_plot, var=args.var)
    print("Training complete!")

if __name__ == '__main__':
    main()
