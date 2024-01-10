import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, epochs, loss_func, train_dl, val_dl, train_size):

    model.to(device)
    out_dict = {'acc_train': [], 'acc_test': [],
                'loss_train': [], 'loss_test': [],
                'f1_train': [], 'f1_test': []}
    
    for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
        model.train()
        train_correct = 0
        train_loss = []
        train_labels, train_preds = [], [] 
        for idx, (data, target) in tqdm(enumerate(train_dl), desc=f"Running batch:", total=len(train_dl)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            train_labels.extend(target.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        model.eval()
        test_loss = []
        test_correct = 0
        test_labels, test_preds = [], []
        for data, target in val_dl:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)

            test_loss.append(loss_func(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            test_labels.extend(target.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

        train_f1 = f1_score(train_labels, train_preds, average='macro')
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        out_dict['acc_train'].append(train_correct/train_size)
        out_dict['acc_test'].append(test_correct/train_size)
        out_dict['loss_train'].append(np.mean(train_loss))
        out_dict['loss_test'].append(np.mean(test_loss))
        out_dict['f1_train'].append(train_f1)
        out_dict['f1_test'].append(test_f1)

    return out_dict, model