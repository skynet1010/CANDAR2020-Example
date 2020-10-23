import torch
from code.utils.consts import loss_dict


def evaluate(
    model: torch.nn.Module,
    eval_data_loader: torch.utils.data.DataLoader,
    criterion:torch.nn.modules.loss._Loss,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype=torch.float
) -> Dict:
    model.to(device=device,dtype=dtype)

    model.eval()

    correct = 0
    total = 0
    running_loss=0

    softmax = torch.nn.Softmax(dim=1)

    tp = 0
    fp = 0

    with torch.no_grad():
        for data in eval_data_loader:
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)
            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"]).to(device=device,dtype=dtype)
            lbl_onehot.zero_()
            lbl_onehot = lbl_onehot.scatter(1,data["labels"].to(device=device,dtype=torch.long),1).to(device=device,dtype=dtype)
            # ===================forward=====================
            
            output = model(img)
            
            out_softmax = softmax(output)

            loss = criterion(out_softmax, lbl_onehot)

            running_loss+=(loss.item()*tmp_batch_size)
            _, predicted = torch.max(out_softmax.data, 1)
            total += tmp_batch_size

            labels = data["labels"].view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

            label_ones_idx = torch.squeeze(labels.nonzero())

            tp_idx = (pred_cpu[label_ones_idx]==labels[label_ones_idx]).nonzero()
            tp += tp_idx.size()[0]

            fp_idx = (pred_cpu[label_ones_idx]!=labels[label_ones_idx]).nonzero()
            fp += fp_idx.size()[0]


    metrics = {"acc":correct/total, "loss":running_loss/total,"true-positive":tp,"false-positive":fp}
    return metrics