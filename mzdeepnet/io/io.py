import pickle
def save(model,filename):
    with open(filename, mode="wb", ) as wfp:
        pickle.dump(model, wfp)
def load(filename):
    with open(filename, mode="rb", ) as rfp:
        model = pickle.load(rfp)
        return model

