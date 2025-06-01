import argparse
import os
from tqdm import tqdm
from PIL import Image
from datasets import Dataset
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import json
from data_utils import cal_fid

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model")

    parser.add_argument('--test_data_path', type=str, default="./dataset/valid/data.arrow", help="The path to the test dataset.")

    parser.add_argument('--VAEwolabel', type=bool, default=True, help="Whether to evaluate VAEwolabel")
    parser.add_argument('--VAEwolabel_output_dir', type=str, default="VAEwolabel/test", help="The path to VAEwolabel's output dir.")

    parser.add_argument('--Genwithlabel', type=bool, default=False, help="Whether to evaluate Genwithlabel")
    parser.add_argument('--Genwithlabel_output_dir', type=str, default="Genwithlabel/test", help="The path to Genwithlabel's output dir.")

    return parser.parse_args()

def grade():
    args = parse_args()

    res = {}
    grd_dataset = Dataset.from_file(args.test_data_path)
    print("********** Grade **********")
    if args.VAEwolabel:
        print("********** VAEwolabel **********")
        recon_path = os.path.join(args.VAEwolabel_output_dir, "recon")
        with open(os.path.join(args.VAEwolabel_output_dir, "hyperparams.json"), 'r') as f:
            hyperparams = json.load(f)
        print("Calculating MSE score and SSIM score ...")
        ssim_score = 0.
        mse_score = 0.
        for i, anc_img in tqdm(enumerate(grd_dataset)):
            anc_img = np.array(anc_img["image"])
            gen_img = np.array(Image.open(os.path.join(recon_path, f"recon_{i}.png")).convert('L'))
            ssim_score += ssim(anc_img, gen_img, data_range=255, channel_axis=None)
            mse_score += mean_squared_error(anc_img / 255., gen_img / 255.)
        ssim_score /= len(grd_dataset)
        mse_score /= len(grd_dataset)
        bonus = 0.2 + 1.0 - hyperparams["latent_dim"]/784 
        score = min((ssim_score * 5 + max(0.1 - mse_score, 0.) * 50) * bonus, 10)
        res["VAEwolabel"] = {"ssim": ssim_score, "mse": mse_score, "score": score}

        print(f"Your SSIM score is: {ssim_score}, MSE score is: {mse_score}. Total score is: {score}")
    if args.Genwithlabel:
        print("********** Genwithlabel **********")
        recon_path = os.path.join(args.Genwithlabel_output_dir, "recon")
        with open(os.path.join(args.Genwithlabel_output_dir, "hyperparams.json"), 'r') as f:
            hyperparams = json.load(f)
        print("Calculating MSE score , SSIM score and FID score ...")
        ssim_score = 0.
        mse_score = 0.
        gen_with_labels = {label: [] for label in range(10)}
        anc_with_labels = {label: [] for label in range(10)}
        for i, anc_data in tqdm(enumerate(grd_dataset)):
            anc_img = np.array(anc_data["image"])
            gen_img = np.array(Image.open(os.path.join(recon_path, f"recon_{i}.png")).convert('L'))
            anc_with_labels[anc_data["label"]].append(anc_img)
            ssim_score += ssim(anc_img, gen_img, data_range=255, channel_axis=None)
            mse_score += mean_squared_error(anc_img / 255., gen_img / 255.)
        ssim_score /= len(grd_dataset)
        mse_score /= len(grd_dataset)

        for label in range(10):
            folder_path = os.path.join(args.Genwithlabel_output_dir, "gen", str(label))
            gen_img_paths = [f for f in os.listdir(folder_path) if f.endswith(".png")]
            for img_path in gen_img_paths:
                gen_img = Image.open(os.path.join(folder_path, img_path)).convert("L")
                gen_with_labels[label].append(np.array(gen_img))   

        fid_score = cal_fid(gen_with_labels, anc_with_labels)
        bonus = 0.2 + 1.0 - hyperparams["latent_dim"]/784 

        score = min((ssim_score * 10 + max(0.1 - mse_score, 0.) * 100 + max(10. - fid_score, 0.)) * bonus, 30)

        res["Genwithlabel"] = {"ssim": ssim_score, "mse": mse_score, "fid": fid_score, "score": score}

        print(f"Your SSIM score is: {ssim_score}, MSE score is: {mse_score}, FID score is: {fid_score}, Total score is: {score}")
    res = json.dumps(res, indent=4)
    with open("grade_result.json", 'w') as file:
        file.write(res)
    print("The result has been saved to grade_result.json!")

if __name__ == "__main__":
    grade()