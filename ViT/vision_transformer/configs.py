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
    }
    
def b16_config():
    config = base_config()
    return config

def b32_config():
    config = b16_config()
    config.update({'patch_size':32})
    return config

def l16_config():
    config = base_config()
    config.update({
        'num_heads':16,
        'num_layers': 24,
        'mlp_size':4096,
        'embed_dim':1024
    })
    return config
    
def l32_config():
    config = l16_config()
    config.update({'patch_size': 32})
    return config
    
def h14_config():
    config = base_config()
    config.update({
        'img_size':392, # img_size 384 doeesn't work
        'patch_size':14,
        'num_heads':16,
        'num_layers': 32,
        'mlp_size':5120,
        'embed_dim':1280
    })
    return config

def mnist_config():
    return {
        'img_channels': 1,
        'img_size': 28,
        'patch_size': 4,
        'num_heads': 4,
        'num_layers': 6,
        'mlp_size': 1024,
        'embed_dim': 64,
        'num_classes': 10,
        'dropout': 0.1
    }
    
def cifar10_config():
    config = mnist_config()
    config.update({
        'img_channels':3,
        'img_size':32
    })
    return config


config_dict = {
    'b16': b16_config(),
    'b32': b32_config(),
    'l16': l16_config(),
    'l32': l32_config(),
    'h14': h14_config(),
}