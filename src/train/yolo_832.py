import wandb
from ultralytics import YOLO
import argparse

def train_yolov11_832(weight_path, num_epochs, wandb_key, data_yaml, dev):
    wandb.login(key=wandb_key)

    model = YOLO(weight_path)
    results = model.train(data=data_yaml, imgsz=832, batch=8, close_mosaic=15, epochs=num_epochs, device=dev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train yolov11 default")
    parser.add_argument(
        "-w",
        "--wandb_key",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
    )
    parser.add_argument(
        "--yaml",
        type=str,
        help="path to yaml file",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="yolov11x.pt"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
    )

    args = parser.parse_args()
    train_yolov11_832(args.weight_path, args.epochs, args.wandb_key, args.yaml, args.device)
