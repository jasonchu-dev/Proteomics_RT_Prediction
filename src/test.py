import torch
import matplotlib.pyplot as plt
from utils import load_model
from dataloader import dataloader
from utils import unnormalize_zero_2_one

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    autoencoder = load_model()
    autoencoder.eval()
    
    train_data, val_data, test_data = dataloader()

    start, end = 0, 100

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for feature, label in test_data:
            y_pred = autoencoder(feature.to(device))
            y_pred = y_pred.squeeze(1, 2)
            y_pred = unnormalize_zero_2_one(y_pred.to('cpu'))
            label = unnormalize_zero_2_one(label)
            predictions.extend(y_pred)
            ground_truth.extend(label)

    plt.figure(figsize=(25, 5))

    plt.plot(predictions[start:end], marker='h', alpha=0.5)
    plt.plot(ground_truth[start:end], marker='h', alpha=0.5)

    plt.show()
    
if __name__ == "__main__":
    main()