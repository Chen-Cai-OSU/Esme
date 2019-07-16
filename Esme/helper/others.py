def filterdict(dict, keys):
    # keys: a list of keys
    dict_ = {}
    for key, val in dict.items():
        if key in keys: dict_[key] = val
    return dict_

def assert_names(keys = [], dict = None):
    # assert all vars in keys exists in the dict
    for key in keys:
        assert key in dict.keys()