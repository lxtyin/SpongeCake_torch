#### SpongeCake_torch

`render.py` 头部，进行场景配置（仅支持渲染平面，单个点光源）

```python
FILM_SIZE = 512                 # Output resolution
PLANE_SIZE = 10                 # Entire fabric size
CAMERA_Z = 5                    # Camera position (0, 0, 8)
FOV = 90                        # Camera fov
LIGHT_POSITION = (0, 0, 7)
LIGHT_INTENSITY = 500
SPP = 4
SCALING = 2
```

渲染用例（运行 `render.py`）：

```python
layer0 = Layer(0.5, 0.5, [0.4, 0.2, 0.7], [1, 1, 1], True)
layer0.set_map("orientation", spectrum2vector(readexr("textures/orientation.exr")), tiles=(80, 80))
layer0.to(DEVICE)

layer1 = Layer(0.3, 1.0, [0.4, 0.2, 0.7], [0.2], False)
layer1.set_map("specular", sRGB2linear(readimg("textures/albedo2.png")))
layer1.to(DEVICE)

rendered = render_twolayer(layer0, layer1)
writeimg("rendered.png", rendered)
```

优化用例（运行 `optimize.py`）

```python
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
```



