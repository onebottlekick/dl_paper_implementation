def base_config():
    return {
        'img_channels': 3,
        'img_size': 224,
        'num_classes': 1000,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_size': 3072,
        'embed_dim':768,
        'dropout': 0.1,
        'kernel_sizes': [7, 3, 3],
        'strides': [4, 2, 2],
        'paddings': [2, 1, 1]
    }


# TODO  -kernel_sizes, strides, paddings
def mnist_config():
    return {
        'img_channels': 1,
        'img_size': 28,
        'num_heads': 4,
        'num_layers': 6,
        'mlp_size': 1024,
        'embed_dim': 64,
        'num_classes': 10,
        'dropout': 0.1
    }
    
def cifar10_config():
    return {
        'img_channels': 3,
        'img_size': 32,
        'num_heads': 8,
        'num_layers': 16,
        'mlp_size': 2048,
        'embed_dim': 128,
        'num_classes': 10,
        'dropout': 0.1
    }
