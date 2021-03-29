from pickle import NONE

from torch.serialization import save
from .imports import *

NO_DEC = ["bias", "BatchNorm2d.weight", "BatchNorm2d.bias", "LayerNorm.weight", 'LayerNorm.bias',
          "BatchNorm1d.weight", "BatchNorm1d.bias"]


def get_hparams(train_dl, model, metric_fc, n_epochs=30, lf=nn.CrossEntropyLoss(), wd=1e-4, no_decay=NO_DEC, opt=torch.optim.AdamW, lr=1e-2):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_transforms = transforms.Compose([transforms.ColorJitter(.3, .3, .3),
                                           transforms.RandomRotation(5),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           normalize
                                           ])

    val_transforms = transforms.Compose([normalize
                                         ])

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

    return n_epochs, lf, params, optimizer, sched, train_transforms, val_transforms


def compute_f1(embeddings, ls, threshold):
    dists = torch.cdist(embeddings, embeddings)
    distances, indices = torch.topk(dists, 50, dim=1, largest=False)

    THRESHOLD = threshold
    preds = [[] for _ in range(embeddings.shape[0])]
    for i in tqdm(range(distances.shape[0]), leave=False):
        IDX = torch.where(distances[i, ] < THRESHOLD)[0]
        IDS = indices[i, IDX]
        preds[i] = IDS.cpu().numpy()

    label_counts = ls.value_counts()
    f_score = 0
    precision = 0
    recall = 0
    for i in tqdm(range(embeddings.shape[0]), leave=False):
        tp = 0
        fp = 0
        true_label = ls.iloc[i]
        for index in preds[i]:
            if ls.iloc[index] == true_label:
                tp += 1
            else:
                fp += 1
        fn = label_counts[true_label] - tp
        # print(label_counts[true_label]-1, tp)
        f_score += 2*tp / (label_counts[true_label] + len(preds[i]))
        precision += tp / len(preds[i])
        recall += tp / label_counts[true_label]
    f_score = f_score/embeddings.shape[0]
    precision = precision/embeddings.shape[0]
    recall = recall/embeddings.shape[0]

    print('f1 score : {} | precision : {} | recall : {}'.format(
        f_score, precision, recall))
    return f_score, precision, recall


def train(model, optimizer, loss_func, sched, metric_fc, train_dl, val_dl, n_epochs, val_df,
          train_transforms, val_transforms, save_path, val_first=False):
    scores = []
    tr_losses = []
    val_scores = []
    prev_best_f_score = -10
    if val_first :
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dl)
            embs = []
            for imgs, _ in pbar:
                imgs = val_transforms(imgs).to('cuda')
                feature = model(imgs)
                embs.append(feature)
            embs = F.normalize(torch.cat(embs, 0))

            f_scores = [compute_f1(embs, val_df['label_group'], thr)
                            for thr in np.linspace(0.2, 1.2, 15)]
            best_f = max(f_scores, key=lambda x: x[0])[0]
            if best_f > prev_best_f_score:
                prev_best_f_score = best_f
                torch.save(model.state_dict(
                ), save_path + '_best.pth'.format(ep))
                print('Saved best model ep {} with f score : {}'.format(
                        -1, best_f))
    for ep in tqdm(range(n_epochs)):
        model.train()
        tr_loss = []
        pbar = tqdm(train_dl)
        for imgs, labels in pbar:

            imgs = train_transforms(imgs).to('cuda')

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

        if ep % 2 == 0:
            torch.save(model.state_dict(
            ), save_path + '_ep_{}.pth'.format(ep))

        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dl)
            embs = []
            for imgs, _ in pbar:
                imgs = val_transforms(imgs).to('cuda')
                feature = model(imgs)
                embs.append(feature)
            embs = F.normalize(torch.cat(embs, 0))

            f_scores = [compute_f1(embs, val_df['label_group'], thr)
                        for thr in np.linspace(0.2, 1.2, 15)]
            best_f = max(f_scores, key=lambda x: x[0])[0]
            if best_f > prev_best_f_score:
                prev_best_f_score = best_f
                torch.save(model.state_dict(
                ), save_path + '_best.pth'.format(ep))
                print('Saved best model ep {} with f score : {}'.format(
                    ep, best_f))

        tr_losses.append(tr_loss)
        val_scores.append(best_f)
        summary = f"Ep {ep}: Train loss {np.asarray(tr_loss).mean()} | Val f score {best_f}"
        print(summary)
    return scores, (tr_losses, val_scores)
