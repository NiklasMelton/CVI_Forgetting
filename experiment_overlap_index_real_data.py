import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from OverlapIndex import OverlapIndex
from iCONN_index import iCONN
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from artlib import normalize
import pickle
from common import make_dirs


BATCH_SIZE = 250
RHO = 0.9
MT = "MT~"
SUB_SAMPLE_SIZE = 1000

def experiment_oi_cnn():

    # ── 1) DEVICE SELECTION ───────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ── 2) SEED ───────────────────────────────────────────────────────
    torch.manual_seed(0)
    np.random.seed(0)

    # ── 3) DATASETS & DATALOADERS ────────────────────────────────────
    transform = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST("./data", train=True, download=True,
                                          transform=transform)
    val_ds = torchvision.datasets.MNIST("./data", train=False, download=True,
                                        transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False,
                            num_workers=4, pin_memory=False)

    # ── 4) LOAD FULL VALIDATION SET ONCE ─────────────────────────────
    val_inputs = []
    val_labels = []
    for xb, yb in val_loader:
        val_inputs.append(xb)
        val_labels.append(yb)
    val_inputs = torch.cat(val_inputs, dim=0).to(torch.float32).to(
        device)  # (10000,1,28,28)
    val_labels = torch.cat(val_labels, dim=0).to(torch.long).to(device)  # (10000,)

    # ── 5) STRATIFIED SUBSAMPLE FOR OI ───────────────────────────────
    y_np = val_labels.cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SUB_SAMPLE_SIZE, random_state=0)
    sub_idx, _ = next(sss.split(np.zeros_like(y_np), y_np))
    sub_inputs = val_inputs[sub_idx]
    sub_labels = val_labels[sub_idx]
    y_sub_np = sub_labels.cpu().numpy()  # for OverlapIndex

    # ── 6) MODEL DEFINITION ───────────────────────────────────────────
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            z1 = self.pool1(x)  # conv1_pooled
            x = F.relu(self.conv2(z1))
            z2 = self.pool2(x)  # conv2_pooled

            flat = z2.view(z2.size(0), -1)
            z3 = F.relu(self.fc1(flat))  # fc1
            out = self.fc2(z3)
            return out, z1, z2, z3

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ── 7) STORAGE ────────────────────────────────────────────────────
    oi_conv1 = []
    oi_conv2 = []
    oi_fc1 = []

    cn_conv1 = []
    cn_conv2 = []
    cn_fc1 = []

    sil_conv1 = []
    sil_conv2 = []
    sil_fc1 = []
    val_accuracy_history = []

    embeddings = []

    # ── 8) TRAINING LOOP ─────────────────────────────────────────────
    model.train()
    for epoch in range(1):  # adjust epochs if needed
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x_batch, y_batch) in loop:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # --- training step ---
            optimizer.zero_grad()
            logits, _, _, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            # --- validation accuracy (every batch) ---
            model.eval()
            with torch.no_grad():
                logits_val, _, _, _ = model(val_inputs)
                preds = logits_val.argmax(dim=1)
                val_acc = (preds == val_labels).float().mean().item()
                val_accuracy_history.append(val_acc)


                # --- only every 5 batches: compute OI on subsample ---
                if (batch_idx % 5 == 0) or (batch_idx in [0, 119, 239]):
                    _, z1_sub, z2_sub, z3_sub = model(sub_inputs)
                    X1 = z1_sub.view(z1_sub.size(0), -1).cpu().numpy()
                    X2 = z2_sub.view(z2_sub.size(0), -1).cpu().numpy()
                    X3 = z3_sub.view(z3_sub.size(0), -1).cpu().numpy()

                    # prepare once per OI instance
                    X1_p, _, _ = normalize(X1)
                    X2_p, _, _ = normalize(X2)
                    X3_p, _, _ = normalize(X3)


                    if batch_idx in [0, 119, 239]:
                        embeddings.append({"X1": X1_p, "X2": X2_p, "X3": X3_p, "y": y_sub_np})

                    if batch_idx % 5 == 0:
                        oi1 = OverlapIndex(rho=RHO, ART="Fuzzy", match_tracking=MT)
                        oi2 = OverlapIndex(rho=RHO, ART="Fuzzy", match_tracking=MT)
                        oi3 = OverlapIndex(rho=RHO, ART="Fuzzy", match_tracking=MT)

                        cn1 = iCONN(rho=RHO, match_tracking=MT)
                        cn2 = iCONN(rho=RHO, match_tracking=MT)
                        cn3 = iCONN(rho=RHO, match_tracking=MT)

                        oi_conv1.append(oi1.add_batch(X1_p, y_sub_np))
                        oi_conv2.append(oi2.add_batch(X2_p, y_sub_np))
                        oi_fc1.append(oi3.add_batch(X3_p, y_sub_np))

                        cn_conv1.append(cn1.add_batch(X1_p, y_sub_np))
                        cn_conv2.append(cn2.add_batch(X2_p, y_sub_np))
                        cn_fc1.append(cn3.add_batch(X3_p, y_sub_np))

                        try:
                            score1 = silhouette_score(X1, y_sub_np)
                        except:
                            score1 = np.nan

                        try:
                            score2 = silhouette_score(X2, y_sub_np)
                        except:
                            score2 = np.nan

                        try:
                            score3 = silhouette_score(X3, y_sub_np)
                        except:
                            score3 = np.nan

                        sil_conv1.append(score1)
                        sil_conv2.append(score2)
                        sil_fc1.append(score3)

            model.train()
            loop.set_postfix(loss=loss.item(), val_acc=val_acc)

    # ── 9) RAW VALIDATION OI ──────────────────────────────────────────
    X_raw = sub_inputs.view(sub_inputs.size(0), -1).cpu().numpy()
    oi_raw = OverlapIndex(rho=RHO, ART="Fuzzy", match_tracking=MT)
    cn_raw = iCONN(rho=RHO, match_tracking=MT)
    X_prep, _, _ = normalize(X_raw)
    oi_val_raw = oi_raw.add_batch(X_prep, y_sub_np)
    cn_val_raw = cn_raw.add_batch(X_prep, y_sub_np)
    try:
        silhouette_raw = silhouette_score(X_raw, y_sub_np)
    except:
        silhouette_raw = np.nan

    print("Done.")
    print(f"Total batches:        {len(train_loader)}")
    print(f"OI computed batches:  {len(oi_conv1)}  (≈ every 5th batch)")
    print(f"Raw-val OI:           {oi_val_raw:.4f}")
    print(f"Raw-val CONN:           {cn_val_raw:.4f}")
    print(f"Raw-val Silhouette:   {silhouette_raw:.4f}")

    path = "results_data/overlap_index/real/mnist_oi.pickle"
    make_dirs(path)
    pickle.dump(
        {
            "oi_conv1_pooled": oi_conv1, "oi_conv2_pooled": oi_conv2, "oi_fc1": oi_fc1,
             "oi_raw": oi_val_raw,
             "cn_conv1_pooled": cn_conv1, "cn_conv2_pooled": cn_conv2, "cn_fc1": cn_fc1,
             "cn_raw": cn_val_raw, "sil_conv1_pooled": sil_conv1,
             "sil_conv2_pooled":sil_conv2, "sil_fc1":sil_fc1, "sil_raw": silhouette_raw,
             "val_accuracy_history": val_accuracy_history,
         },
        open(path, "wb")
    )
    path = "results_data/overlap_index/real/mnist_embeddings.pickle"
    make_dirs(path)
    pickle.dump(
        embeddings,
        open(path, "wb")
    )

if __name__ == "__main__":
    experiment_oi_cnn()
