import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
from tqdm import tqdm

from OFI import OFI
from common import make_dirs
from ActivationKNN import KNN


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        z1 = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(z1)))
        x = F.relu(self.bn4(self.conv4(x)))
        z2 = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(z2)))
        x = F.relu(self.bn6(self.conv6(x)))
        z3 = self.pool3(x)

        flat = z3.view(z3.size(0), -1)
        x = F.relu(self.fc1(flat))
        out = self.fc2(x)

        return out, z1, z2, z3


def run_condition_cnn(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        batch_size: int = 50,
        sub_batch_size: int = 5,
        rho: float = 0.9,
        r_hat: float = 0.1,
        ART: str = "Fuzzy",
        is_ordered: bool = False
) -> Tuple[List[List[float]], List[float], List[Tuple[float, float]], List[float]]:
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    cf_detector = OFI(rho=rho, r_hat=r_hat, ART=ART, match_tracking="MT~")

    tpr_trace: List[List[float]] = []
    oi_trace: List[float] = []
    ocf_trace: List[Tuple[float, float]] = []
    val_accuracy_trace: List[float] = []

    y_test_np = y_test.numpy()
    train_ds = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,
                          drop_last=True, shuffle=False)

    model.train()
    last_class_seen = -1
    for x_b, y_b in tqdm(train_ds):
        x_b, y_b = x_b.to(device), y_b.to(device)

        if is_ordered:
            # Determine the dominant class in the batch
            current_class = y_b[0].item()  # assuming ordered and pure batches
            if current_class != last_class_seen:
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=1e-3)  # reset optimizer
                last_class_seen = current_class

        prev_loss = float("inf")
        for _ in range(10):  # set a max to avoid infinite loops
            model.train()
            out, *_ = model(x_b)
            loss = criterion(out, y_b)

            # Check convergence
            if abs(prev_loss - loss.item()) < 1e-4:
                break
            prev_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2) eval on full test set
        model.eval()
        with torch.no_grad():
            logits_test = []
            correct = 0
            total = 0
            for i in range(0, len(X_test), 256):
                xb = X_test[i:i + 256].to(device)
                logits, *_ = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds.cpu() == y_test[i:i + 256]).sum().item()
                total += preds.size(0)
                logits_test.append(logits.cpu())
            logits = torch.cat(logits_test).numpy()
            val_accuracy = correct / total
            val_accuracy_trace.append(val_accuracy)
        model.train()

        # 3) TPR computation
        y_pred = logits.argmax(axis=1)
        tprs = []
        for cls in range(10):
            mask = y_test_np == cls
            tp = np.sum(y_pred[mask] == cls)
            fn = np.sum(y_pred[mask] != cls)
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        tpr_trace.append(tprs)

        indices = np.random.choice(batch_size, sub_batch_size, replace=False)

        # 4) OCF update with subsampled data
        O, F = cf_detector.add_batch(
            X_train=x_b[indices].cpu().numpy().reshape(sub_batch_size, -1),
            y_train=y_b[indices].cpu().numpy(),
            y_pred_eval=y_pred,
            y_true_eval=y_test_np,
            y_scores_eval=logits
        )
        ocf_trace.append((O, F))
        oi_trace.append(cf_detector.OI.index)

    return tpr_trace, oi_trace, ocf_trace, val_accuracy_trace


def expiriment_cnn():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True,
                                transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True,
                               transform=transform)

    X_train_all = torch.tensor(train_ds.data).permute(0, 3, 1, 2).float() / 255.0
    y_train_all = torch.tensor(train_ds.targets)
    X_test = torch.tensor(test_ds.data).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(test_ds.targets)

    traces_tpr: Dict[str, List[List[float]]] = {}
    traces_oi: Dict[str, List[float]] = {}
    traces_state: Dict[str, List[Tuple[float, float]]] = {}
    val_acc_trace: Dict[str, List[float]] = {}

    for order in ("Shuffled", "Ordered"):
        if order == "Shuffled":
            idx = torch.randperm(len(y_train_all))
        else:
            idx = torch.argsort(y_train_all)

        X_train = X_train_all[idx]
        y_train = y_train_all[idx]

        print(f"→ Running CIFAR condition: {order}")
        tpr, oi, states, val_acc = run_condition_cnn(
            X_train, y_train, X_test, y_test,
            batch_size=250,
            sub_batch_size=25,
            rho=0.9,
            r_hat=np.inf,
            ART="Fuzzy",
            is_ordered=(order == "Ordered")
        )

        traces_tpr[order] = tpr
        traces_oi[order] = oi
        traces_state[order] = states
        val_acc_trace[order] = val_acc

    path = "results_data/OFI/real/ofi_traces_cifar_cnn.npz"
    make_dirs(path)
    np.savez(
        path,
        ordered_tpr=traces_tpr["Ordered"],
        shuffled_tpr=traces_tpr["Shuffled"],
        ordered_int=traces_oi["Ordered"],
        shuffled_int=traces_oi["Shuffled"],
        ordered_states=np.array(traces_state["Ordered"]),
        shuffled_states=np.array(traces_state["Shuffled"]),
        ordered_accuracy=val_acc_trace["Ordered"],
        shuffled_accuracy=val_acc_trace["Shuffled"]
    )

    print("Saved CIFAR CNN traces with OFI, Overlap Index, and per-batch accuracy.")


def run_condition_knn(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    batch_size: int = 50,
    sub_batch_size: int = 5,
    rho: float = 0.9,
    r_hat: float = 0.1,
    ART: str   = "Fuzzy"
) -> Tuple[List[List[float]], List[float], List[Tuple[float, float]], List[float]]:

    clf = KNN(n_neighbors=1)
    cf_detector = OFI(rho=rho, r_hat=r_hat, ART=ART, match_tracking="MT~")

    tpr_trace: List[List[float]] = []
    oi_trace:  List[float] = []
    ocf_trace: List[Tuple[float, float]] = []
    val_accuracy_trace: List[float] = []

    y_test_np = y_test.numpy()
    X_test_np = X_test.reshape(len(X_test), -1).numpy()

    # Incremental "fitting" by re-training KNN on accumulating data
    X_seen = []
    y_seen = []

    train_ds = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, drop_last=True, shuffle=False)

    for x_b, y_b in tqdm(train_ds):
        x_b_np = x_b.reshape(batch_size, -1).numpy()
        y_b_np = y_b.numpy()

        # Accumulate training data
        X_seen.append(x_b_np)
        y_seen.append(y_b_np)

        X_seen_np = np.vstack(X_seen)
        y_seen_np = np.concatenate(y_seen)

        # Train KNN on accumulated data
        clf.fit(X_seen_np, y_seen_np)

        # 1) evaluate
        y_pred = clf.predict(X_test_np)
        val_accuracy = np.mean(y_pred == y_test_np)
        val_accuracy_trace.append(val_accuracy)

        # 2) compute TPR
        tprs = []
        for cls in range(10):
            mask = y_test_np == cls
            tp = np.sum(y_pred[mask] == cls)
            fn = np.sum(y_pred[mask] != cls)
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        tpr_trace.append(tprs)

        indices = np.random.choice(batch_size, sub_batch_size, replace=False)

        # 3) compute OCF indices
        y_scores = clf.activation(X_test_np)
        O, F = cf_detector.add_batch(
            X_train       = x_b_np[indices].reshape(sub_batch_size, -1),
            y_train       = y_b_np[indices],
            y_pred_eval   = y_pred,
            y_true_eval   = y_test_np,
            y_scores_eval = y_scores
        )
        ocf_trace.append((O, F))
        oi_trace.append(cf_detector.OI.index)

    return tpr_trace, oi_trace, ocf_trace, val_accuracy_trace


def experiment_knn():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    X_train_all = torch.tensor(train_ds.data).permute(0, 3, 1, 2).float() / 255.0
    y_train_all = torch.tensor(train_ds.targets)
    X_test      = torch.tensor(test_ds.data).permute(0, 3, 1, 2).float() / 255.0
    y_test      = torch.tensor(test_ds.targets)

    traces_tpr:     Dict[str, List[List[float]]]         = {}
    traces_oi:      Dict[str, List[float]]               = {}
    traces_state:   Dict[str, List[Tuple[float, float]]] = {}
    val_acc_trace:  Dict[str, List[float]]               = {}

    for order in ("Shuffled", "Ordered"):
        if order == "Shuffled":
            idx = torch.randperm(len(y_train_all))
        else:
            idx = torch.argsort(y_train_all)

        X_train = X_train_all[idx]
        y_train = y_train_all[idx]

        print(f"→ Running CIFAR condition: {order}")
        tpr, oi, states, val_acc = run_condition_knn(
            X_train, y_train, X_test, y_test,
            batch_size=250,
            sub_batch_size = 25,
            rho=0.9,
            r_hat=np.inf,
            ART="Fuzzy"
        )

        traces_tpr[order]    = tpr
        traces_oi[order]     = oi
        traces_state[order]  = states
        val_acc_trace[order] = val_acc

    path = "results_data/OFI/real/ofi_traces_cifar_knn.npz"
    make_dirs(path)
    np.savez(
        path,
        ordered_tpr       = traces_tpr["Ordered"],
        shuffled_tpr      = traces_tpr["Shuffled"],
        ordered_int       = traces_oi["Ordered"],
        shuffled_int      = traces_oi["Shuffled"],
        ordered_states    = np.array(traces_state["Ordered"]),
        shuffled_states   = np.array(traces_state["Shuffled"]),
        ordered_accuracy  = val_acc_trace["Ordered"],
        shuffled_accuracy = val_acc_trace["Shuffled"]
    )

    print("Saved CIFAR KNN traces with OFI, Overlap Index, and per-batch accuracy.")
