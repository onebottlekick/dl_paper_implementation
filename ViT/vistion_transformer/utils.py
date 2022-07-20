def pair(x):
    return x if isinstance(x, tuple) else (x, x)


def build_model(model, config):
    model = model(**config)
    return model