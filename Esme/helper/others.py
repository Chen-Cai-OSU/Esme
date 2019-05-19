def filterdict(dict, keys):
    # keys: a list of keys
    dict_ = {}
    for key, val in dict.items():
        if key in keys: dict_[key] = val
    return dict_
