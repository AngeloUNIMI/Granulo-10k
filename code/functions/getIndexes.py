from sklearn.model_selection import KFold

def getIndexes(num_iterations, r, train_index, seed_adams):

    # indexes train val
    kf_train_val = KFold(n_splits=num_iterations-1, shuffle=True, random_state=seed_adams+r)
    train_temp, val_temp = next(kf_train_val.split(train_index))
    train_index_2 = train_index[train_temp]
    val_index = train_index[val_temp]

    # test_indices = indices[r*fold_size: (r+1)*fold_size]
    # train_indices = np.concatenate([indices[:r*fold_size], indices[(r+1)*fold_size:]])

    return train_index_2, val_index
