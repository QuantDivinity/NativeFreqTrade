
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Attempt to import BaseFreqaiModel and FreqaiDataKitchen from freqtrade
try:
    from freqtrade.freqai.base_models import BaseFreqaiModel
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
except ImportError:
    logging.warning(
        "Could not import Freqtrade specific modules. Ensure this model is used within a Freqtrade environment and freqtrade is installed correctly."
    )

# Define dummy classes if running standalone for basic syntax checking
class BaseFreqaiModel:
    def __init__(self, **kwargs):
        self.freqai_info = kwargs.get('freqai_info', {})
        self.ft_config = self.freqai_info.get('feature_engineering', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_split_parameters = self.freqai_info.get('data_split_parameters', {})
        self.model_training_parameters = self.freqai_info.get('model_training_parameters', {})
        self.live_retrain_hours = self.model_training_parameters.get('live_retrain_hours')


class FreqaiDataKitchen:
    pass


# Logger
logger = logging.getLogger(__name__)


# RltvMSELoss class
class RltvMSELoss(torch.nn.Module):
    def __init__(self):
        super(RltvMSELoss, self).__init__()

    def forward(self, yhat, y):
        return torch.mean((yhat - y) ** 2) / torch.mean(y ** 2) + 1e-6


# GeneralTSNet class
class GeneralTSNet(nn.Module):
    def __init__(
        self,
        feats_num: int,
        hidden_num: int,
        droprate1: float = 0,
        droprate2: float = 0,
        input_bn: bool = True,
        out_fc_item_size: int = 1,
        attention_layer: nn.Module = None,
        out_num: int = 1
    ):
        super(GeneralTSNet, self).__init__()

        self.bn1 = None
        if input_bn:
            self.bn1 = nn.BatchNorm1d(feats_num)
        self.dropout1 = nn.Dropout(droprate1)
        self.gru = nn.GRU(feats_num, hidden_num, 1, batch_first=True, dropout=0.0)
        self.attention = attention_layer
        if attention_layer is not None:
            hidden_num = hidden_num * 2
        self.bn2 = nn.BatchNorm1d(hidden_num)
        self.dropout2 = nn.Dropout(droprate2)

        if out_num == 1:
            if out_fc_item_size > 1:
                self.fc = nn.Sequential(
                    nn.Linear(hidden_num, out_fc_item_size),
                    nn.ReLU(),
                    nn.Linear(out_fc_item_size, 1)
                )
            else:
                self.fc = nn.Linear(hidden_num, 1, bias=True)
        else:
            self.fc = nn.Linear(hidden_num, out_num, bias=True)

    def forward(self, x, train_mode=None):
        if torch.isnan(x).any():
            logger.error("NaN detected in input tensor to GeneralTSNet")
            raise ValueError("NaN detected in input tensor to GeneralTSNet")

        if self.bn1 is not None:
            x = torch.transpose(x, 1, 2)
            x = self.bn1(x)
            x = torch.transpose(x, 1, 2)
        x = self.dropout1(x)
        x, _ = self.gru(x, None)
        gru_out = x[:, -1, :]
        if self.attention is not None:
            att_out = self.attention(x)
            x2 = torch.cat([gru_out, att_out], dim=1)
        else:
            x2 = gru_out

        if torch.isnan(x2).any():
            logger.error("NaN detected in tensor x2 (after GRU/Attention) in GeneralTSNet")
            raise ValueError("NaN detected in tensor x2 (after GRU/Attention) in GeneralTSNet")

        x2 = self.dropout2(x2)
        out = self.fc(x2)
        return out.squeeze(1) if out.size(1) == 1 else out

    def reinitialize(self):
        for name, p in self.named_parameters():
            if 'lstm' in name or 'gru' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def reinitialize2(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# GRUFreqaiModel class
class GRUFreqaiModel(BaseFreqaiModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_params = self.freqai_info.get('model_params', {})
        self.batch_size = self.model_params.get('batch_size', 64)
        self.learning_rate = self.model_params.get('learning_rate', 1e-3)
        self.epochs = self.model_params.get('epochs', 50)
        self.feats_num = self.ft_config.get('num_features', 5)
        self.hidden_num = self.model_params.get('hidden_num', 32)
        self.out_fc_item_size = self.model_params.get('out_fc_item_size', 1)
        self.dropout1 = self.model_params.get('dropout1', 0.0)
        self.dropout2 = self.model_params.get('dropout2', 0.0)
        self.input_bn = self.model_params.get('input_bn', True)
        self.model = GeneralTSNet(
            feats_num=self.feats_num,
            hidden_num=self.hidden_num,
            droprate1=self.dropout1,
            droprate2=self.dropout2,
            input_bn=self.input_bn,
            out_fc_item_size=self.out_fc_item_size,
            out_num=1
        )
        self.model.to(self.device)
        self.criterion = RltvMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info(f"GRUFreqaiModel initialized. Features: {self.feats_num}, Hidden: {self.hidden_num}, Device: {self.device}")

    def train(self, data: np.ndarray, labels: np.ndarray, weights: np.ndarray, dk: FreqaiDataKitchen, **kwargs) -> Any:
        normalized_data = np.copy(data).astype(np.float32)
        for i in range(data.shape[0]):
            first_open = data[i, 0, 0]
            if first_open != 0 and not np.isnan(first_open):
                normalized_data[i, :, :] /= (first_open + 1e-6)
            else:
                logger.warning(f"Sample {i} in training data has first_open='{first_open}', skipping normalization.")

        x_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.model.train()

        logger.info(f"Starting GRU model training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            processed_batches = 0
            for i, (batch_x, batch_y) in enumerate(train_loader): # Inner loop for batches
                if batch_x.shape[0] == 0:
                    logger.warning(f"Skipping empty batch at epoch {epoch + 1}, batch index {i}")
                    continue
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                if predictions.shape != batch_y.shape:
                    logger.error(f"Shape mismatch! Predictions: {predictions.shape}, Batch Y: {batch_y.shape}")
                    if predictions.numel() != batch_y.numel(): # Check if total elements match
                        try:
                            predictions = predictions.view_as(batch_y)
                        except RuntimeError as e:
                             raise ValueError(f"Unrecoverable shape mismatch and view_as failed. Pred: {predictions.shape}, Labels: {batch_y.shape}. Error: {e}")
                    else:
                        # This case (numel matches but shape doesn't, and view_as would fail) is less common for 1D outputs.
                        # If batch_y is (N, 1) and predictions is (N), then unsqueeze predictions.
                        if predictions.ndim == batch_y.ndim -1 and predictions.shape[0] == batch_y.shape[0] and batch_y.shape[1] == 1:
                            predictions = predictions.unsqueeze(1)
                            if predictions.shape != batch_y.shape: # Double check after potential fix
                                raise ValueError(f"Shape mismatch after attempting unsqueeze. Pred: {predictions.shape}, Labels: {batch_y.shape}")
                        else:
                            raise ValueError(f"Unrecoverable shape mismatch. Pred: {predictions.shape}, Labels: {batch_y.shape}")

                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                processed_batches += 1
            # End of inner loop (batch loop)

            # This block should be at the same indentation level as the inner loop's definition
            # and epoch_loss = 0.0, processed_batches = 0
            if processed_batches > 0:
                avg_epoch_loss = epoch_loss / processed_batches
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                    logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Avg Loss: {avg_epoch_loss:.4f}")
            else:
                logger.warning(f"Epoch {epoch + 1} had no processed batches. Check data and DataLoader.")
        return self.model

    def predict(self, data: np.ndarray, dk: FreqaiDataKitchen, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        normalized_data = np.copy(data).astype(np.float32)
        for i in range(data.shape[0]):
            first_open = data[i, 0, 0]
            if first_open != 0 and not np.isnan(first_open):
                normalized_data[i, :, :] /= (first_open + 1e-6)
            else:
                logger.warning(f"Sample {i} in prediction data has first_open='{first_open}', using unnormalized.")
        
        x_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_tensor)
        
        di_values = predictions.cpu().numpy()
        if di_values.ndim == 0:
            di_values = np.array([di_values])
        
        user_defined_predictions = np.zeros((data.shape[0], 0))
        return di_values, user_defined_predictions
