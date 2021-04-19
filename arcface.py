from imports import *


# https://github.com/ronghuaiyang/arcface-pytorch

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, centers=None,
                 device='cuda', half=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.device = device
        if centers is None:
            print('Using random weights')
            self.weight = Parameter(
                torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)
        else:
            print('Using center as wieghts')
            self.weight = Parameter(centers.to(device))
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device=device)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output


def compute_centers(dataloader, model, val_transforms, dataframe, batch_size=64):
    dataframe['label_group'] = dataframe['label_group'].astype(
        'category').cat.codes
    dataframe['indx'] = range(len(dataframe))
    label_indexes = dataframe.groupby('label_group').agg({'indx': 'unique'})
    with torch.no_grad():
        embs = []
        for imgs, _ in tqdm(dataloader):
            imgs = val_transforms(imgs).to('cuda')
            features = model(imgs)
            embs.append(features)
    embs = F.normalize(torch.cat(embs, 0))
    centers = torch.zeros(len(label_indexes), embs.shape[1]).to('cuda')
    for i in range(len(label_indexes)):
        centers[i] = embs[label_indexes.iloc[i].values[0]].mean(dim=0)
    return centers
