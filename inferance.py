import torch
from calflops import calculate_flops
from numpy import mean
from torchvision import models


def cal_flops():
    from archs.CFD.fc1 import fc1
    model = torch.load("./saves/fc1/CFD/9_model_lt.pth.tar")
    batch_size = 1
    input_shape = (batch_size, 20)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    # Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M


def cal_time():
    import time
    # from archs.CFD.fc1 import fc1
    from archs.cosmoflow.fc1 import fc1, get_standard_cosmoflow_model
    model = get_standard_cosmoflow_model().to("cuda:0")
    # model = torch.load("./saves/fc1/cosmoflow/0_model_lt.pth.tar")
    for name, param in model.named_parameters():
        print(name)
    batch_size = 1
    input_shape = (batch_size, 20)
    input_shape = (batch_size, 4, 128, 128, 128)
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(13):
            x = torch.randn(input_shape).to("cuda:0")
            start = time.time()
            y = model(x)
            end = time.time()
            times.append(end - start)
    print("Time cost: %s" % mean(times[3:]))

    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


if __name__ == "__main__":
    # cal_flops()
    # cal_time()
    import tensorflow as tf
    # inspect cuda device
    print(tf.config.list_physical_devices('GPU'))

