from pickle import NONE
from re import S
import copy
import matplotlib.pyplot as plt

from torch.serialization import save
from .imports import *

NO_DEC = ["bias", "BatchNorm2d.weight", "BatchNorm2d.bias", "LayerNorm.weight", 'LayerNorm.bias',
          "BatchNorm1d.weight", "BatchNorm1d.bias"]


def get_tfms(resize=None, crop=224):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_transforms = transforms.Compose([transforms.ColorJitter(.3, .3, .3),
                                           transforms.RandomRotation(5),
                                           transforms.RandomCrop(crop),
                                           transforms.RandomHorizontalFlip(),
                                           normalize
                                           ])
    val_tfms = [normalize]
    if resize is not None :
        val_tfms  = [transforms.Resize((resize,resize))]
    val_transforms = transforms.Compose(val_tfms)
    return train_transforms, val_transforms


def get_hparams(train_dl, model, metric_fc, n_epochs=30, lf=nn.CrossEntropyLoss(), wd=1e-4, no_decay=NO_DEC, opt=torch.optim.AdamW, lr=1e-2,
                param_groups=None):

    params = list(model.named_parameters()) + \
        list(metric_fc.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = opt(optimizer_grouped_parameters, lr=lr)

    # learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.3,  # anneal_strategy = 'linear',
                                                total_steps=int(n_epochs * len(train_dl)))

    return n_epochs, lf, params, optimizer, sched


def compute_f1(embeddings, ls, thresholds):
    dists = torch.cdist(embeddings, embeddings)

    distances, indices = torch.topk(dists, 50, dim=1, largest=False)
    scores = {}
    best_thr = None
    best_score = -10
    for threshold in tqdm(thresholds, leave=False, desc='Thresholds'):
        THRESHOLD = threshold
        preds = [[] for _ in range(embeddings.shape[0])]
        for i in range(distances.shape[0]):
            IDX = torch.where(distances[i, ] < THRESHOLD)[0]
            IDS = indices[i, IDX]
            preds[i] = IDS.cpu().numpy()

        label_counts = ls.value_counts()
        f_score = 0
        for i in range(embeddings.shape[0]):
            tp = 0
            fp = 0
            true_label = ls.iloc[i]
            for index in preds[i]:
                if ls.iloc[index] == true_label:
                    tp += 1
                else:
                    fp += 1
            fn = label_counts[true_label] - tp
            #print(label_counts[true_label]-1, tp)
            f_score += 2*tp / (label_counts[true_label] + len(preds[i]))
        f_score = f_score/embeddings.shape[0]
        if f_score > best_score:
            best_thr = threshold
            best_score = f_score
        scores[threshold] = f_score

    return scores, best_thr, best_score


def train_full_data(model, optimizer, loss_func, sched, metric_fc, train_dl, n_epochs, train_df,
          train_transforms, val_transforms, save_path, val_first=False, 
          prev_best_info={'train': {'thr': None, 'f1': None}},
          info_history=[], ep_start=0):

    tr_losses = []
    tr_scores = []
    prev_best_f_score = -10

    for ep in tqdm(range(ep_start, ep_start + n_epochs), leave=False):
        # TRAINING
        model.train()
        tr_loss = []
        embs = []
        ys = []
        pbar = tqdm(train_dl, leave=False)
        for imgs, labels in pbar:
            ys.append(labels)
            imgs = train_transforms(imgs.to('cuda'))

            optimizer.zero_grad()
            feature = model(imgs)
            labels = labels.long().to('cuda')
            out = metric_fc(feature, labels)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()
            sched.step()

            tr_loss.append(loss.item())
            pbar.set_description(f"Train loss: {round(np.mean(tr_loss),3)}")
            embs.append(feature.detach())
        ys = pd.Series(torch.cat(ys, 0).numpy())
        embs = F.normalize(torch.cat(embs, 0))
        
        # compute fsccores
        if prev_best_info['train']['thr'] is None :
            thrs = np.linspace(0.2, 1, 10)
        else :
            thrs = thrs = [prev_best_info['train']['thr'] - 0.1, prev_best_info['train']['thr'] - 0.05, prev_best_info['train']['thr'], prev_best_info['train']['thr'] + 0.05, prev_best_info['train']['thr'] + 0.1]
        train_f1s, best_thresh_tr, f1_tr = compute_f1(embs, ys, thrs)
        prev_best_info['train']['thr'], prev_best_info['train']['f1'] = best_thresh_tr, f1_tr

        if ep % 2 == 0:
            path =  save_path + '_ep_{}.pth'.format(ep)
            print('Checkpoint : saved model to {}'.format(path))
            torch.save(model.state_dict(
            ),path)

        info_history.append(copy.deepcopy(prev_best_info))

        tr_losses.append(tr_loss)
        summary = "Ep {}: Loss {:.4f} | F score {:.4f} with thresh {:.2f}".format(
            ep, np.asarray(tr_loss).mean(),  f1_tr, best_thresh_tr)
        print(summary)
    return prev_best_info, info_history, (tr_losses)

def train(model, optimizer, loss_func, sched, metric_fc, train_dl, val_dl, n_epochs, train_df, val_df,
          train_transforms, val_transforms, save_path, val_first=False, 
          prev_best_info={'val': {'thr': None, 'f1': None}, 'train': {'thr': None, 'f1': None}},
          info_history=[], ep_start=0):

    tr_losses = []
    tr_scores = []
    val_scores = []
    prev_best_f_score = -10

    for ep in tqdm(range(ep_start, ep_start + n_epochs), leave=False):
        # TRAINING
        model.train()
        tr_loss = []
        embs = []
        ys = []
        pbar = tqdm(train_dl, leave=False)
        for imgs, labels in pbar:
            ys.append(labels)
            imgs = train_transforms(imgs.to('cuda'))

            optimizer.zero_grad()
            feature = model(imgs)
            labels = labels.long().to('cuda')
            out = metric_fc(feature, labels)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()
            sched.step()

            tr_loss.append(loss.item())
            pbar.set_description(f"Train loss: {round(np.mean(tr_loss),3)}")
            embs.append(feature.detach())
        ys = pd.Series(torch.cat(ys, 0).numpy())
        embs = F.normalize(torch.cat(embs, 0))
        
        # compute fsccores
        if prev_best_info['train']['thr'] is None :
            thrs = np.linspace(0.2, 1, 10)
        else :
            thrs = [prev_best_info['train']['thr'] - 0.1, prev_best_info['train']['thr'] - 0.05, prev_best_info['train']['thr'], prev_best_info['train']['thr'] + 0.05, prev_best_info['train']['thr'] + 0.1]
        train_f1s, best_thresh_tr, f1_tr = compute_f1(embs, ys, thrs)
        prev_best_info['train']['thr'], prev_best_info['train']['f1'] = best_thresh_tr, f1_tr

        if ep % 2 == 0:
            path =  save_path + '_ep_{}.pth'.format(ep)
            print('Checkpoint : saved model to {}'.format(path))
            torch.save(model.state_dict(
            ),path)

        # VALIDATION
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dl, leave=False)
            embs = []
            for imgs, _ in pbar:
                imgs = val_transforms(imgs).to('cuda')
                feature = model(imgs)
                embs.append(feature)
            embs = F.normalize(torch.cat(embs, 0))

            # compute fsccores
            if prev_best_info['val']['thr'] is None :
                thrs = np.linspace(0.2, 1, 10)
            else :
                thrs = [prev_best_info['val']['thr'] - 0.1, prev_best_info['val']['thr'] - 0.05, prev_best_info['val']['thr'], prev_best_info['val']['thr'] + 0.05, prev_best_info['val']['thr'] + 0.1]
            val_f1s, best_thresh_val, f1_val = compute_f1(embs, val_df['label_group'], thrs)
            prev_best_info['val']['thr'], prev_best_info['val']['f1'] = best_thresh_val, f1_val

            if f1_val > prev_best_f_score:
                prev_best_f_score = f1_val
                torch.save(model.state_dict(
                ), save_path + '_best.pth'.format(ep))
                print('Saved best model ep {} with f score : {}'.format(
                    ep, f1_val))
        info_history.append(copy.deepcopy(prev_best_info))

        tr_losses.append(tr_loss)
        val_scores.append(f1_val)
        summary = "Ep {}: Train loss {:.4f} | Val f score {:.4f} with thresh {:.2f}, train f score {:.4f} with thresh {:.2f}".format(
            ep, np.asarray(tr_loss).mean(), f1_val, best_thresh_val, f1_tr, best_thresh_tr)
        print(summary)
    return prev_best_info, info_history, (tr_losses, val_scores)

def freeze_resnet_to(model, layer) :
    layers = ['layer{}'.format(i) for i in range(0, layer+1)]
    for n, p in model.named_parameters() :
        if any(layer in n for layer in layers) :
            p.requires_grad = False
def unfreeze_resnet_to(model, layer) :
    layers = ['layer{}'.format(i) for i in range(layer, 5)]
    for n, p in model.named_parameters() :
        if any(layer in n for layer in layers) :
            p.requires_grad = True
def freeze(model) :
    for p in model.parameters() :
        p.requires_grad = False

def unfreeze(model) :
    for p in model.parameters() :
        p.requires_grad = True

def plot_hist(history) : 
    train_scores = [info['train']['f1'] for info in history]
    train_thr = [info['train']['thr'] for info in history]
    val_scores = [info['val']['f1'] for info in history]
    val_thr = [info['val']['thr'] for info in history]
    x = range(len(train_scores))
    plt.plot(x, val_scores, label='val_score')
    plt.plot(x, train_scores, label='train_score')
    plt.legend()
    plt.show()
    plt.plot(x, val_thr, label='val_thr')
    plt.plot(x, train_thr, label='train_thr')
    plt.legend()
    plt.show()
