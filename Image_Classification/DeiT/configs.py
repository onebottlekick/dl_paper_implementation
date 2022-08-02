def base_config():
    return {
        'img_channels': 3,
        'img_size': 224,
        'num_classes': 1000,
        'patch_size': 16,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_size': 3072,
        'embed_dim':768,
        'dropout': 0.1,
        'token_type':'cls+distil'
    }
    
    
def t16_config():
    config = base_config()
    config.update({
        'embed_dim':192,
        'num_heads':3
    })
    return config

def b16_config():
    config = base_config()
    return config

def mnist_config():
    config = base_config()
    config.update({
        'img_channels':1,
        'img_size':28,
        'num_classes':10,
        'patch_size':4,
        'num_heads':4,
        'num_layers':6,
        'mlp_size':1024,
        'embed_dim':64
    })
    return config

config_dict = {
    't16': t16_config(),
    'b16': b16_config()
}