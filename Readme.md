#### SpongeCake_torch

`render.py` 头部，进行场景配置（仅支持渲染平面，单个点光源）

```python
FILM_SIZE = 512                             # Output resolution
PLANE_SIZE = 10                             # Entire fabric size
CAMERA_UP = (0, 1, 0)						# Camera up direction
CAMERA_LOOK_AT = ((0, 0, 5), (0, 0, 0))     # (origin, target)
FOV = 90                                    # Camera fov
LIGHT_POSITION = (0, 0, 7)
LIGHT_INTENSITY = 400
SPP = 1
SCALING = 2
```

层级设置、渲染用例（运行 `render.py`）：

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

优化用例见 `optimize.py`



