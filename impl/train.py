import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def calculate_auc(preds, labels):
    if labels.unique().shape[0] == 2:
        preds = torch.sigmoid(preds)
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    else:
        preds = torch.softmax(preds, dim=-1)
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')

def random_mask_selection_with_ratio(train_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Randomly selects a subset of `True` elements from a boolean tensor `train_mask`
    based on a specified ratio.

    Args:
        train_mask (torch.Tensor): A boolean tensor.
        ratio (float): The ratio of `True` elements to select (between 0 and 1).

    Returns:
        torch.Tensor: A new boolean tensor with the same shape as `train_mask`,
                      where `ceil(ratio * num_true_elements)` elements are randomly set to `True`.
    """
    # Validate the ratio
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")
    
    # Get indices of True elements
    true_indices = torch.nonzero(train_mask, as_tuple=True)[0]
    num_true_elements = len(true_indices)
    
    # Calculate the number of elements to select
    num_select = max(1, int(ratio * num_true_elements))  # At least 1 element if ratio > 0
    
    # Randomly select indices
    selected_indices = true_indices[torch.randperm(num_true_elements)[:num_select]]
    
    # Create a new mask
    new_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    new_mask[selected_indices] = True
    
    return new_mask


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


def train(optimizer, model, dataloader, metrics, loss_fn, device, row, col, run, epoch):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    ys = []
    preds = []
    set_seed(epoch + run * 1000)
    row = row.to(device)
    col = col.to(device)
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], row=row, col=col, device=device, id=0)
        loss = loss_fn(pred, batch[-1])
        preds.append(pred)
        ys.append(batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.detach().cpu().numpy(), y.detach().cpu().numpy()), sum(total_loss) / len(total_loss)


def train_model(optimizer, model, G, features, sparse_adj, metrics):
    '''
    Train models in an epoch.
    '''
    model.train()
    # for batch in dataloader:
    optimizer.zero_grad()
    train_mask = G.mask==0
    loss = model.loss(features, sparse_adj, G.y[train_mask], train_mask)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        pred = model.predict(features, sparse_adj, train_mask)
        auc = calculate_auc(pred, G.y[train_mask])
        f1 = metrics(pred.cpu().numpy(), G.y[train_mask].cpu().numpy())
    return f1, auc, loss.item()



@torch.no_grad()
def test_model(model, G, features, sparse_adj, metrics, testing=True):
    '''
    Train models in an epoch.
    '''
    model.eval()
    if testing:
        mask = G.mask==2
    else:
        mask = G.mask==1
    target = G.y[mask]
    pred = model.predict(features, sparse_adj, mask)
    f1 = metrics(pred.cpu().numpy(), target.cpu().numpy())
    auc = calculate_auc(pred, target)
    return f1, auc



@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, device, row, col, run, epoch):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    set_seed(epoch + run * 1000)
    row = row.to(device)
    col = col.to(device)
    for batch in dataloader:
        pred = model(*batch[:-1], row=row, col=col, device=device)
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)