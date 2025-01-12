from utils import *
from sggx import FiberLikeSGGX, SurfaceLikeSGGX
import torch
import math

# region configuration

FILM_SIZE = 512                             # Output resolution
PLANE_SIZE = 10                             # Entire fabric size
CAMERA_UP = (0, 1, 0)                       # Camera up direction
CAMERA_LOOK_AT = ((0, 0, 5), (0, 0, 0))     # (origin, target)
FOV = 90                                    # Camera fov
LIGHT_POSITION = (0, 0, 7)
LIGHT_INTENSITY = 400
SPP = 1
SCALING = 2

# endregion


# Intersect with the fabric plane.
# Same as mitsuba, wi points to the camera and wo points to the light.
# Return: wo_to_light, 1.0/distance_to_light,
#         wi_to_camera, u, v, points. (all of shape(spp, height, width, channel))
def get_intersections_plus(spp=SPP):
    assert (int(math.sqrt(spp)) ** 2 == spp)

    # camera space
    zz = -FILM_SIZE * 0.5 * math.tan(FOV * PI / 360)
    x = torch.arange(0.0, FILM_SIZE, 1.0).to(DEVICE) - FILM_SIZE / 2
    y = -torch.arange(0.0, FILM_SIZE, 1.0).to(DEVICE) + FILM_SIZE / 2
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    root = torch.sqrt(torch.as_tensor(spp))
    root = torch.round(root).int()
    interval = 1 / root
    xs = []
    ys = []
    for i in range(root):
        for j in range(root):
            x_offset = i * interval + torch.rand([FILM_SIZE, FILM_SIZE]).to(DEVICE) / root
            y_offset = j * interval + torch.rand([FILM_SIZE, FILM_SIZE]).to(DEVICE) / root
            xs.append(grid_x + x_offset)
            ys.append(grid_y - y_offset)

    grid_x = torch.stack(xs, dim=0)
    grid_y = torch.stack(ys, dim=0)

    wis = torch.stack([grid_x, grid_y, zz * torch.ones_like(grid_x)], dim=-1)
    wis = normalize3(wis)

    # transform to world space
    origin = torch.tensor(CAMERA_LOOK_AT[0], dtype=FLOAT_TYPE).to(DEVICE)
    target = torch.tensor(CAMERA_LOOK_AT[1], dtype=FLOAT_TYPE).to(DEVICE)
    up = torch.tensor(CAMERA_UP, dtype=FLOAT_TYPE).to(DEVICE)

    forward = normalize3(target - origin)
    right = normalize3(torch.cross(up, -forward))
    up = normalize3(torch.cross(-forward, right))
    rotation = torch.stack([right, up, -forward], dim=1)

    assert (not torch.allclose(right, torch.zeros_like(right)))

    wis = wis.view(-1, 3)
    wis = normalize3(torch.matmul(wis, rotation.T).view(spp, FILM_SIZE, FILM_SIZE, 3))

    # intersection
    time = -origin[..., [2]] / wis[..., [2]]
    position = origin + time * wis

    uvs = position[..., [0, 1]] / PLANE_SIZE + 0.5
    # uvs[uvs > 1] = 0.0
    # uvs[uvs < 0] = 0.0
    us = uvs[..., [0]]
    vs = uvs[..., [1]]
    wos = torch.tensor(LIGHT_POSITION, dtype=FLOAT_TYPE).to(DEVICE) - position
    idis = 1.0 / length(wos)
    wos = normalize3(wos)

    return (wos, idis, -wis, us, vs, position)


set_seed(0) # for default intersection
default_intersections = get_intersections_plus()


class Layer:
    def __init__(self, roughness = 0.1, thickness = 1.0, specular = [1, 1, 1], F0 = [1, 1, 1], fiberlike = False):
        self.roughness = torch.tensor([roughness], dtype=FLOAT_TYPE)
        self.thickness = torch.tensor([thickness], dtype=FLOAT_TYPE)
        self.specular = torch.tensor(specular, dtype=FLOAT_TYPE)
        self.F0 = torch.tensor(F0, dtype=FLOAT_TYPE)
        self.maps = {}
        self.maptiles = {}
        self.sggx = FiberLikeSGGX() if fiberlike else SurfaceLikeSGGX()

    def set_map(self, name, img, tiles=(1, 1)):
        # img: tensor(h, w, c)
        self.maps[name] = img
        self.maptiles[name] = tiles

    def set_requires_grad(self, tf):
        ls = self.opt_params()
        for i in ls:
            i.requires_grad = tf

    def to(self, devic):
        self.roughness = self.roughness.to(devic)
        self.thickness = self.thickness.to(devic)
        self.specular = self.specular.to(devic)
        self.F0 = self.F0.to(devic)
        for k, v in self.maps.items():
            self.maps[k] = v.to(devic)

    def opt_params(self):
        varlist = [self.roughness,
                   self.thickness,
                   self.specular,
                   self.F0]
        for k, v in self.maps.items():
            varlist.append(v)
        return varlist

    def param_name(self):
        def rstr(x, rd=2):
            if len(x.shape) == 0:
                return round(x.item(), rd)
            else:
                s = ""
                for i in x:
                    s = s + f",{round(i.item(), rd)}"
                return s[1:]


        name = (f"R_{rstr(self.roughness)}_T_{rstr(self.thickness)}"
                f"Ks_{rstr(self.specular, 3)}_F0_{rstr(self.F0)}")
        return name

    def correct_params(self):
        with torch.no_grad():
            self.roughness.data = torch.clip(self.roughness, 0.01, 1)
            self.thickness.data = torch.clip(self.thickness, 0.01, 5)
            self.specular.data = torch.clip(self.specular, 0, 1)
            self.F0.data = torch.clip(self.F0, 0, 1)
            # TODO correct other values in your maps
            if "roughness" in self.maps.keys():
                self.maps["roughness"].data = torch.clip(self.maps["roughness"], 0.01, 1)
            if "thickness" in self.maps.keys():
                self.maps["thickness"].data = torch.clip(self.maps["thickness"], 0.01, 1)
            if "specular" in self.maps.keys():
                self.maps["specular"].data = torch.clip(self.maps["specular"], 0, 1)
            if "F0" in self.maps.keys():
                self.maps["F0"].data = torch.clip(self.maps["F0"], 0, 1)

    def get_svdata(self, u, v, name, default_value = torch.tensor([0.0])):
        if name not in self.maps.keys():
            return torch.tile(default_value, u.shape)
        else:
            img = self.maps[name]
            tiles = self.maptiles[name]
            u = u * tiles[0] % 1
            v = v * tiles[1] % 1
            xindexs = (u * img.shape[1]).long()[..., 0]
            yindexs = ((1 - v) * img.shape[0]).long()[..., 0]
            xindexs[xindexs == img.shape[1]] = img.shape[1] - 1
            yindexs[yindexs == img.shape[0]] = img.shape[0] - 1
            return img[yindexs, xindexs]

    def get_all_svdata(self, u, v):
        normal = self.get_svdata(u, v, "normal", torch.tensor([0, 0, 1.0])).to(DEVICE)
        orientation = self.get_svdata(u, v, "orientation", torch.tensor([0, 0, 1.0])).to(DEVICE)
        thickness = self.get_svdata(u, v, "thickness", self.thickness).to(DEVICE)
        roughness = self.get_svdata(u, v, "roughness", self.roughness).to(DEVICE)
        specular = self.get_svdata(u, v, "specular", self.specular).to(DEVICE)
        F0 = self.get_svdata(u, v, "F0", self.F0).to(DEVICE)

        normal = normalize3(normal)
        orientation = normalize3(orientation)

        thickness[torch.logical_or(u > 1, u < 0)] = 0.0
        thickness[torch.logical_or(v > 1, v < 0)] = 0.0

        # all info our bsdf needs.
        return normal, orientation, roughness, thickness, specular, F0


    def specular_fresnel_schlick(self, f0, ldoth):
        return torch.maximum(f0 + (1 - f0) * torch.pow(1 - ldoth, 5), torch.zeros_like(f0))

    # return shape: (spp, h, w, 3)
    def eval(self, wi, wo, us, vs):
        result = torch.zeros_like(wi)
        normal, orientation, roughness, thickness, specular, F0 = self.get_all_svdata(us, vs)
        # normal unused

        # left = torch.cross(orientation, normal)
        # t_wi = to_local(wi, normal, left, orientation)
        # t_wo = to_local(wo, normal, left, orientation)

        s, t = build_orthbasis(orientation)
        t_wi = to_local(wi, s, t, orientation)
        t_wo = to_local(wo, s, t, orientation)
        t_half = normalize3(t_wi + t_wo)
        t_half[length(t_half)[..., 0] == 0] = torch.tensor([0, 0, 1.0]).to(DEVICE)

        sigmaTi = self.sggx.sigmaT(t_wi, roughness)
        sigmaTo = self.sggx.sigmaT(t_wo, roughness)

        i_cosi = wi[..., [2]]
        i_coso = wo[..., [2]]
        i_cosi[i_cosi == 0] = 1
        i_coso[i_coso == 0] = 1

        i_cosi = 1.0 / i_cosi
        i_coso = 1.0 / i_coso

        Ai = sigmaTi * i_cosi
        Ao = sigmaTo * i_coso
        E0, E1 = torch.zeros_like(Ai), torch.zeros_like(Ao)
        E0[Ai <= 0] += Ai[Ai <= 0]
        E0[Ao <= 0] += Ao[Ao <= 0]
        E1[Ai > 0] += Ai[Ai > 0]
        E1[Ao > 0] += Ao[Ao > 0]
        iAio = Ai + Ao
        iAio[iAio == 0] = 1
        iAio = 1.0 / iAio

        D = self.sggx.D(t_half, roughness)
        G = (torch.exp(thickness * E0) - torch.exp(-thickness * E1)) * iAio
        F = self.specular_fresnel_schlick(F0, torch.maximum(dot(t_half, t_wi), torch.zeros_like(us)))

        result += F * specular * D * 0.25 * G * abs(i_cosi)

        if not torch.all(result >= 0):
            raise
        return result * SCALING


    # return shape: (spp, h, w, 1).
    def eval_attenuation(self, ws, us, vs):

        normal, orientation, roughness, thickness, specular, F0 = self.get_all_svdata(us, vs)

        result = torch.zeros_like(roughness)

        s, t = build_orthbasis(orientation)
        t_w = to_local(ws, s, t, orientation)
        sigmaT = self.sggx.sigmaT(t_w, roughness)
        i_wdn = torch.abs(ws[..., [2]])
        condition = (i_wdn != 0)

        i_wdn[i_wdn == 0] = 1
        i_wdn = 1.0 / i_wdn

        result[condition] = torch.exp(-thickness * sigmaT * i_wdn)[condition]

        if not torch.all(result >= 0):
            raise
        return result


# Return both reflection and transmission render results.
def render_twolayer(layer0, layer1, intersections = default_intersections):
    WOs, IDISs, WIs, Us, Vs, POINTs = intersections

    if LIGHT_POSITION[2] > 0:
        # reflection

        specular0 = layer0.eval(WIs, WOs, Us, Vs)
        specular1 = layer1.eval(WIs, WOs, Us, Vs)

        attenuation = (layer0.eval_attenuation(WOs, Us, Vs) *
                       layer0.eval_attenuation(WIs, Us, Vs))

        result = specular0 + specular1 * attenuation

        result = result * LIGHT_INTENSITY * IDISs * IDISs
        result = torch.sum(result, dim=0) / SPP
        del specular0, specular1, attenuation
    else:
        # transmission

        specular0 = layer0.eval(WIs, WOs, Us, Vs)
        specular1 = layer1.eval(WIs, WOs, Us, Vs)

        attenuation0 = layer1.eval_attenuation(WOs, Us, Vs)
        attenuation1 = layer0.eval_attenuation(WIs, Us, Vs)

        result = specular0 * attenuation0 + specular1 * attenuation1

        result = result * LIGHT_INTENSITY * IDISs * IDISs
        result = torch.sum(result, dim=0) / SPP
        del specular0, specular1, attenuation0, attenuation1

    if not torch.all(result >= 0):
        raise

    result = torch.minimum(result, torch.tensor(1.0))
    result = linear2sRGB(result)
    return result


def render_singlelayer(layer0):
    WOs, IDISs, WIs, Us, Vs, POINTs = default_intersections

    result = layer0.eval(WIs, WOs, Us, Vs)
    result = result * LIGHT_INTENSITY * IDISs * IDISs
    result = torch.sum(result, dim=0) / SPP

    if not torch.all(result >= 0):
        raise

    result = torch.minimum(result, torch.tensor(1.0))
    result = linear2sRGB(result)
    return result


if __name__ == "__main__":

    layer0 = Layer(0.3, 0.2, [0.4, 0.2, 0.7], [1, 1, 1], False)
    # layer0.set_map("orientation", spectrum2vector(readexr("textures/orientation.exr")), tiles=(80, 80))
    layer0.to(DEVICE)

    layer1 = Layer(0.6, 100.0, [0.4, 0.2, 0.7], [0.2], False)
    layer1.set_map("specular", sRGB2linear(readimg("textures/albedo2.png")))
    layer1.to(DEVICE)

    rendered = render_twolayer(layer0, layer1)
    # rendered = render_singlelayer(layer0)

    writeimg("rendered.png", rendered)




