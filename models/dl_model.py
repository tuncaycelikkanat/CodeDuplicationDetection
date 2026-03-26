import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
import os

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# Optimize matrix multiplications for AVX/VNNI DL Boost
torch.set_float32_matmul_precision('high')


class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # nn.Sigmoid() -> BCEWithLogitsLoss includes sigmoid
        )

    def forward(self, x):
        return self.net(x).squeeze()


class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.is_sparse = sp.issparse(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert this row to a dense matrix only when requested. Memory friendly.
        x_val = self.X[idx].toarray()[0] if self.is_sparse else self.X[idx]
        x_tensor = torch.tensor(x_val, dtype=torch.float32)

        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor
        return x_tensor,


class PyTorchClassifier:
    def __init__(self, random_state=42, epochs=5, batch_size=256, lr=0.001, device="cpu"):
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # Standard device mapping
        if device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "xpu":
            self.device = torch.device('xpu' if HAS_IPEX and hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
            
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        
        dataset = SparseDataset(X, y)
        
        # Data loading optimization
        n_workers = min(4, os.cpu_count() or 1)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=n_workers, pin_memory=True, drop_last=False
        )

        input_dim = X.shape[1]
        self.model = SimpleMLP(input_dim).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"\n[PyTorch] Training on device: {self.device} (IPEX: {HAS_IPEX})")
        
        # 2. XPU (Intel Arc) Acceleration & Mixed Precision
        if HAS_IPEX and self.device.type == 'xpu':
            self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer, dtype=torch.bfloat16)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                if self.device.type == 'xpu':
                    # Using mixed precision (BF16) -> Significantly reduces training time
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        self.model.eval()
        
        dataset = SparseDataset(X)
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=min(4, os.cpu_count() or 1), pin_memory=True
        )
        
        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                if self.device.type == 'xpu':
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                    
                preds = torch.sigmoid(outputs) > 0.5
                predictions.append(preds.cpu().numpy().astype(int))

        return np.concatenate(predictions)

    def predict_proba(self, X):
        """Return probability estimates for each class [P(0), P(1)]."""
        self.model.eval()

        dataset = SparseDataset(X)
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=min(4, os.cpu_count() or 1), pin_memory=True
        )

        probas = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                if self.device.type == 'xpu':
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                pos_proba = torch.sigmoid(outputs).cpu().numpy()
                probas.append(pos_proba)

        pos = np.concatenate(probas)
        return np.column_stack([1 - pos, pos])

    def get_params(self, deep=True):
        """Return model parameters (sklearn-compatible interface)."""
        return {
            'random_state': self.random_state,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': str(self.device),
        }


def build_dl_model(random_state=42, device="cpu"):
    return PyTorchClassifier(random_state=random_state, device=device)
