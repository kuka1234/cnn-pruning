
# Run network slimming (training base model with network slimming, pruning with network slimming, and fine-tuning to recover accuracy)

import subprocess

def train_model(args):
    subprocess.run([
        "python", 
        "main.py"
    ] + args)

def prune_model(args):
    subprocess.run([
        "python", 
        "prune.py"
    ] + args)


def run_experiments(pruning_method, percentage):
    prune_model([
        pruning_method, 
        "random", 
        f"--model={weights_path}/base_run.pth.tar",
        f"--save={weights_path}",
        f"--percent={percentage}",
        "--dataset=cifar100"    
    ])

    train_model([
        f"fine_tune_{pruning_method}_random",
        f"--save={weights_path}",
        f"--fine_tuning={weights_path}/base_run.pth_pruned_{pruning_method}_random.pth.tar",
        f"--info={percentage}_cifar100",
    ] + additional_args)

if __name__ == "__main__":
    weights_path = f"./model_weights/network_slimming"
    additional_args = ["--simple-classifier", "-sr", f"--s=0.0001", "--dataset=cifar100"]

    # train_model([
    #     "base_run", 
    #     f"--save={weights_path}"
    # ] + additional_args)

    for percentage in [0.5, 0.9, 0.95, 0.99]:
        run_experiments("network_slimming", percentage)