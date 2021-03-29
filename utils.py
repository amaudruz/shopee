from imports import *

def compute_f1(embeddings, ls, threshold) :
    dists = torch.cdist(embeddings, embeddings)
    distances, indices = torch.topk(dists, 50, dim=1, largest=False)
    
    THRESHOLD = threshold
    preds = [[] for _ in range(embeddings.shape[0])]
    for i in tqdm(range(distances.shape[0]), leave=False) :
        IDX = torch.where(distances[i,]<THRESHOLD)[0]
        IDS = indices[i,IDX]
        preds[i] = IDS.cpu().numpy()
            
    label_counts = ls.value_counts()
    f_score = 0 
    precision = 0
    recall = 0
    for i in tqdm(range(embeddings.shape[0]), leave=False) :
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
        precision += tp / len(preds[i])
        recall += tp/ label_counts[true_label]
    f_score = f_score/embeddings.shape[0]
    precision = precision/embeddings.shape[0]
    recall = recall/embeddings.shape[0]
    
    print('f1 score : {} | precision : {} | recall : {}'.format(f_score, precision, recall))
    return f_score, precision, recall

def load_data(df_path='data/train.csv', train_perc=0.7) :
    # load in data

    df = pd.read_csv(df_path)
    labels = np.random.permutation(df['label_group'].unique())

    train_perc = 0.7
    train_idx = int(train_perc * len(labels))

    train_labels = labels[:train_idx]
    val_labels = labels[train_idx:]

    train_df = df[df['label_group'].isin(train_labels)]
    val_df = df[df['label_group'].isin(val_labels)]  

    return df, train_df, val_df, train_labels, val_labels