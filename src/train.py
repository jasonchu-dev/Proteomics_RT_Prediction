import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Encoder, Decoder, peptide2RT
from dataloader import dataloader
from utils import format_number

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    train_data, val_data, test_data = dataloader()

    with open('configs/hyperparameters.yaml', 'r') as file:
        hyperparameters = yaml.safe_load(file)

    reduction = hyperparameters['criterion']['reduction']
    delta = hyperparameters['criterion']['delta']
    lr = hyperparameters['optimizer']['lr']
    weight_decay = hyperparameters['optimizer']['weight_decay']
    mode = hyperparameters['scheduler']['mode']
    factor = hyperparameters['scheduler']['factor']
    patience = hyperparameters['scheduler']['patience']
    min_lr = hyperparameters['scheduler']['min_lr']
    epochs = hyperparameters['epochs']['epochs']

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = peptide2RT(encoder, decoder).to(device)
    criterion = nn.HuberLoss(reduction=reduction, delta=delta)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=True, min_lr=min_lr)

    torch.cuda.empty_cache()

    for epoch in range(1, epochs+1):
        autoencoder.train()
        training_losses = []
        scaled_train_losses = []
        for feature, label in train_data:
            optimizer.zero_grad()
            y_pred = autoencoder(feature.to(device))
            y_pred = y_pred.squeeze(1, 2)
            training_loss = criterion(y_pred, label.to(device))
            scaled_train_loss = criterion(y_pred * 20000, (label * 20000).to(device))
            # training_loss.backward()
            scaled_train_loss.backward()
            optimizer.step()
            training_losses.extend([training_loss.item()])
            scaled_train_losses.extend([scaled_train_loss.item()])

        val_losses = []
        scaled_val_losses = []
        with torch.no_grad():
            for feature, label in val_data:
                y_pred = autoencoder(feature.to(device))
                y_pred = y_pred.squeeze(1, 2)
                val_loss = criterion(y_pred, label.to(device))
                scaled_val_loss = criterion(y_pred * 20000, (label * 20000).to(device))
                val_losses.extend([val_loss.item()])
                scaled_val_losses.extend([scaled_val_loss.item()])

        scheduler.step(scaled_train_loss)

        training_avg_loss = sum(training_losses) / len(training_losses)
        scaled_train_avg_loss = sum(scaled_train_losses) / len(scaled_train_losses)
        val_avg_loss = sum(val_losses) / len(val_losses)
        scaled_val_avg_loss = sum(scaled_val_losses) / len(scaled_val_losses)

        epoch = str(epoch).rjust(5)
        training_avg_loss = format_number(training_avg_loss)
        scaled_train_avg_loss = format_number(scaled_train_avg_loss)
        val_avg_loss = format_number(val_avg_loss)
        scaled_val_avg_loss = format_number(scaled_val_avg_loss)


        print(f'Epoch: {epoch} | Train Loss: {training_avg_loss} | Scaled Train Loss: {scaled_train_avg_loss} | Val Loss: {val_avg_loss} | Scaled Val Loss: {scaled_val_avg_loss}')

if __name__ == "__main__":
    main()