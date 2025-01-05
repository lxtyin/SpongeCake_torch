from render import *
from torch import nn

if __name__ == "__main__":

    layer0 = Layer(0.4, 1.6, [0.1, 0.5, 0.7], [1, 1, 1], True)
    layer0.set_map("orientation", spectrum2vector(readexr("textures/orientation.exr")), tiles=(80, 80))
    layer0.to(DEVICE)

    layer1 = Layer(0.9, 1.0, [0.4, 0.2, 0.7], [0.2, 0.2, 0.2], False)
    layer1.set_map("specular", sRGB2linear(readimg("textures/albedo1.png")))
    layer1.to(DEVICE)

    layer0.set_requires_grad(True)
    layer1.set_requires_grad(True)

    optimizer = torch.optim.Adam(layer0.opt_params() + layer1.opt_params(), lr=0.01, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [50, 100, 150, 200, 250],
                                                     gamma=0.5,
                                                     last_epoch=-1)

    target = readimg("textures/target.png").to(DEVICE)

    for i in range(101):
        rendered = render_twolayer(layer0, layer1)
        if i % 10 == 0:
            print(f"Epoch: {i}")
            writeimg(f"optimize/epoch{i}.png", rendered)
        loss = nn.L1Loss()(rendered, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        layer0.correct_params()
        layer1.correct_params()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

    layer0.set_requires_grad(False)
    layer1.set_requires_grad(False)

    writeimg("optimize/optimized_albedo.png", linear2sRGB(layer1.maps["specular"]))

    rendered = render_twolayer(layer0, layer1)
    writeimg("optimize/rendered.png", rendered)
