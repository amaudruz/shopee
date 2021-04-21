from imports import *
import copy


NO_DEC = ["bias", "BatchNorm2d.weight", "BatchNorm2d.bias", "LayerNorm.weight", 'LayerNorm.bias',
          "BatchNorm1d.weight", "BatchNorm1d.bias"]
NORMALIZE_DEF = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
CLIP_NORMALIZE = transforms.Normalize(mean=(
    0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
normalize_dic = {'timm': NORMALIZE_DEF, 'clip': CLIP_NORMALIZE}


def get_tfms(resize=None, crop=224, normalize='timm'):

    normalize = normalize_dic[normalize]
    train_transforms = transforms.Compose([transforms.ColorJitter(.3, .3, .3),
                                           transforms.RandomRotation(5),
                                           transforms.RandomCrop(crop),
                                           transforms.RandomHorizontalFlip(),
                                           normalize
                                           ])
    val_tfms = [normalize]
    if resize is not None:
        val_tfms = [transforms.Resize((resize, resize))]
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


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def convert_models_to_fp32_(model):
    for p in model.parameters():
        p.data = p.data.float()


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