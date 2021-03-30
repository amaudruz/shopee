from pickle import NONE

from torch.serialization import save
from .imports import *

NO_DEC = ["bias", "BatchNorm2d.weight", "BatchNorm2d.bias", "LayerNorm.weight", 'LayerNorm.bias',
          "BatchNorm1d.weight", "BatchNorm1d.bias"]
          
          
def text_to_device(text, device):
    return {'input_ids' : text['input_ids'].to(device),
            'attention_mask' : text['attention_mask'].to(device)}


def get_hparams(train_dl, model, metric_fc, n_epochs=30, lf=nn.CrossEntropyLoss(), wd=1e-4, no_decay=NO_DEC, opt=torch.optim.AdamW, lr=1e-2, head_only = False):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_transforms = transforms.Compose([transforms.ColorJitter(.3, .3, .3),
                                           transforms.RandomRotation(5),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           normalize
                                           ])

    val_transforms = transforms.Compose([transforms.Resize((224,224)),
                                         normalize
                                         ])
    if head_only:
        params = list(model.head.named_parameters()) + list(metric_fc.named_parameters())
    else:
        params = list(model.named_parameters())  + list(metric_fc.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": wd,
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = opt(optimizer_grouped_parameters, lr=lr)

    # learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.3,  # anneal_strategy = 'linear',
                                                total_steps=int(n_epochs * len(train_dl)))

    return n_epochs, lf, params, optimizer, sched, train_transforms, val_transforms



def compute_f1(embeddings, ls, thresholds) :
    dists = torch.cdist(embeddings, embeddings)
    
    distances, indices = torch.topk(dists, 50, dim=1, largest=False)
    scores = {}
    for threshold in thresholds:
        THRESHOLD = threshold
        preds = [[] for _ in range(embeddings.shape[0])]
        for i in range(distances.shape[0]) :
            IDX = torch.where(distances[i,]<THRESHOLD)[0]
            IDS = indices[i,IDX]
            preds[i] = IDS.cpu().numpy()

        label_counts = ls.value_counts()
        f_score = 0 
        for i in range(embeddings.shape[0]) :
            tp = 0
            fp = 0
            true_label = ls.iloc[i]
            for index in preds[i] :
                if ls.iloc[index] == true_label :
                    tp += 1
                else :
                    fp += 1
            fn = label_counts[true_label] - tp
            #print(label_counts[true_label]-1, tp)
            f_score += 2*tp / (label_counts[true_label] + len(preds[i]))
        f_score = f_score/embeddings.shape[0]
        scores[threshold] = f_score
    return scores


def train(model, optimizer, loss_func, sched, metric_fc, tr_dl, val_dl, n_epochs, val_df,
          train_transforms, val_transforms, save_path, val_first=False):
    device = torch.device('cuda')
    best_thresh_tr = 0.5
    scores = []
    tr_losses = []
    val_scores = []
    prev_best_f_score = -10
    if val_first :
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dl)
            embs = []
            for image, text, label in pbar:

                x = train_transforms(image.to(device)), text_to_device(text, device)
                label = label.to(device)
                feature = model(x)
                embs.append(feature)
            embs = F.normalize(torch.cat(embs, 0))

            f_scores = compute_f1(embs, ys, [i/10 for i in range(4,12)])
            best_f = max(f_scores, key=f_scores.get)
            if best_f > prev_best_f_score:
                prev_best_f_score = best_f
                torch.save(model.state_dict(
                ), save_path + '_best.pth'.format(ep))
                print('Saved best model ep {} with f score : {}'.format(
                        -1, best_f))
    for ep in tqdm(range(n_epochs)):
        model.train()
        tr_loss = []
        pbar = tqdm(tr_dl)
        embs = []
        ys = []
        for image, text, label in pbar:
            ys.append(label)
        
            x = train_transforms(image.to(device)), text_to_device(text, device)
            label = label.to(device)
        
            optimizer.zero_grad()
            feature = model(x)
            embs.append(feature)
            out = metric_fc(feature, label)
            loss = loss_func(out, label) #+ lf2(anchor_emb, pos_emb, neg_emb)
            
            loss.backward()
            optimizer.step()
            sched.step()
        
            tr_loss.append(loss.item())
            pbar.set_description(f"Train loss: {round(np.mean(tr_loss),3)}")
        ys = pd.Series(torch.cat(ys, 0).numpy())    
        embs = F.normalize(torch.cat(embs, 0))
        f1s = compute_f1(embs, ys, [best_thresh_tr - 0.1, best_thresh_tr, best_thresh_tr + 0.1])
        best_thresh_tr = max(f1s, key=f1s.get)
        f1_tr = f1s[best_thresh_tr]   
        
        if ep % 2 == 0:
            torch.save(model.state_dict(
            ), save_path + '_ep_{}.pth'.format(ep))

        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dl)
            embs = []
            for image, text, label in pbar:

                x = val_transforms(image.to(device)), text_to_device(text, device)
                label = label.to(device)
                feature = model(x)
                embs.append(feature)
            embs = F.normalize(torch.cat(embs, 0))

            f_scores = compute_f1(embs, val_df['label_group'], [best_thresh_tr - 0.1, best_thresh_tr, best_thresh_tr + 0.1])
            best_t = max(f_scores, key=f_scores.get)
            best_f = f_scores[best_t]
            if best_f > prev_best_f_score:
                prev_best_f_score = best_f
                torch.save(model.state_dict(
                ), save_path + '_best.pth'.format(ep))
                print('Saved best model ep {} with f score : {}'.format(
                    ep, best_f))

        tr_losses.append(tr_loss)
        val_scores.append(best_f)
        summary = f"Ep {ep}: Train loss {np.asarray(tr_loss).mean()} | Val f score {best_f} with thresh {best_t}, train f score {f1_tr} with thresh {best_thresh_tr}"
        print(summary)
    return scores, (tr_losses, val_scores)
