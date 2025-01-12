import torch

import render
from render import *
from torch import nn

if __name__ == "__main__":

    # target

    target_layer0 = Layer(0.5, 0.2, [0.4, 0.2, 0.7], [1, 1, 1], False)
    target_layer0.to(DEVICE)

    target_layer1 = Layer(0.6, 100.0, [0.4, 0.2, 0.7], [0.2], False)
    target_layer1.set_map("specular", sRGB2linear(readimg("textures/albedo2.png")))
    target_layer1.set_map("F0", sRGB2linear(readimg("textures/albedo2.png")))
    target_layer1.to(DEVICE)

    # current

    layer0 = Layer(0.4, 0.3, [0.1, 0.5, 0.7], [1, 1, 1], False)
    layer0.to(DEVICE)

    layer1 = Layer(0.9, 2.0, [0.4, 0.2, 0.7], [0.2, 0.2, 0.2], False)
    layer1.set_map("specular", torch.ones(128, 128, 3, dtype=FLOAT_TYPE))
    layer1.set_map("F0", torch.ones(128, 128, 3, dtype=FLOAT_TYPE))
    layer1.to(DEVICE)

    layer0.set_requires_grad(True)
    layer1.set_requires_grad(True)

    optimizer = torch.optim.Adam(layer0.opt_params() + layer1.opt_params(), lr=0.01, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [25, 50, 100, 150],
                                                     gamma=0.5,
                                                     last_epoch=-1)

    light_pos_list = [(0, 0, 7), (0, 6, 7), (6, 0, 7), (4, 4, 7)]

    for i in range(len(light_pos_list)):
        render.LIGHT_POSITION = light_pos_list[i]
        intersections = get_intersections_plus(1)
        target = render_twolayer(target_layer0, target_layer1, intersections)
        writeimg(f"optimize/target_{i}.png", target)

    param_list = []
    for i in range(204):
        render.LIGHT_POSITION = light_pos_list[i % len(light_pos_list)]
        intersections = get_intersections_plus(1)
        target = render_twolayer(target_layer0, target_layer1, intersections)
        rendered = render_twolayer(layer0, layer1, intersections)

        loss = nn.L1Loss()(rendered, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        layer0.correct_params()
        layer1.correct_params()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        if (i // 4) % 5 == 0:
            print(f"Epoch: {i}_{i % 4}, Loss: {loss}")
            writeimg(f"optimize/epoch{i}_{i % 4}.png", rendered)
            print(layer0.param_name(), layer1.param_name())
        param_list.append((i, layer0.param_name(), layer1.param_name()))

    layer0.set_requires_grad(False)
    layer1.set_requires_grad(False)

    writeimg("optimize/optimized_albedo.png", linear2sRGB(layer1.maps["specular"]))
    with open(os.path.join('optimize/params.txt'), 'w') as f:
        for i in param_list:
            f.write(str(i) + '\n')
