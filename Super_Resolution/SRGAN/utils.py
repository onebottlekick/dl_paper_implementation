def calc_shape(input_size, kernel_size=3, stride=2, padding=1, iter_num=4):
    def _calc_shape(input_size, kernel_size, stride, padding):
        h, w = input_size
        h = (h + 2*padding - kernel_size)//stride + 1
        w = (w + 2*padding - kernel_size)//stride + 1
        return h, w
    for _ in range(iter_num):
        input_size = _calc_shape(input_size, kernel_size, stride, padding)
    return input_size