from utils import *
from sggx import FiberLikeSGGX, SurfaceLikeSGGX
import torch
import math

# region configuration

FILM_SIZE = 512                 # Output resolution
PLANE_SIZE = 10                 # Entire fabric size
CAMERA_Z = 5                    # Camera position (0, 0, 8)
FOV = 90                        # Camera fov
LIGHT_POSITION = (0, 0, 7)
LIGHT_INTENSITY = 500
SPP = 4
SCALING = 2

# endregion


# Intersect with the fabric plane.
# Same as mitsuba, wi points to the camera and wo points to the light.
# Return: wo_to_front_light, wo_to_back_light, 1.0/distance_to_front_light, 1.0/distance_to_back_light,
#         wi_to_camera, u, v, points. (all of shape(spp, height, width, channel))
def get_intersections(spp=SPP):

    V_PLANE_SIZE = 2 * CAMERA_Z * math.tan(FOV * PI / 360)    # Visible fabric size
    CAMERA_POS  = torch.tensor([V_PLANE_SIZE / 2, -V_PLANE_SIZE / 2, CAMERA_Z])
    LIGHT_POS = torch.tensor([V_PLANE_SIZE / 2 + LIGHT_POSITION[0],
                              -V_PLANE_SIZE / 2 + LIGHT_POSITION[1],
                              LIGHT_POSITION[2]])
    assert (V_PLANE_SIZE <= PLANE_SIZE)
    assert (int(math.sqrt(spp)) ** 2 == spp)

    root = torch.sqrt(torch.as_tensor(spp))
    root = torch.round(root).int()
    interval = 1 / root
    points = []
    for i in range(root):
        for j in range(root):
            x_offset = i * interval + torch.rand([FILM_SIZE, FILM_SIZE]) / root
            y_offset = j * interval + torch.rand([FILM_SIZE, FILM_SIZE]) / root

            offset = torch.stack([x_offset, -y_offset], dim=2)
            dx = torch.arange(0, FILM_SIZE)
            dy = torch.arange(0, -FILM_SIZE, -1).reshape(FILM_SIZE, 1)
            offset[..., 0] += dx
            offset[..., 1] += dy
            offset *= V_PLANE_SIZE / FILM_SIZE
            pos = torch.concat([offset, torch.zeros(FILM_SIZE, FILM_SIZE, 1)], dim=2)
            points.append(pos)

    points = torch.stack(points)
    uvs = points[..., [0, 1]] / PLANE_SIZE
    uvs[..., 1] += 1
    wo = LIGHT_POS - points
    idis = 1.0 / length(wo)
    wo = normalize3(wo)
    wi = normalize3(CAMERA_POS - points)
    us = uvs[..., [0]]
    vs = uvs[..., [1]]
    return (wo.to(DEVICE), idis.to(DEVICE),
            wi.to(DEVICE), us.to(DEVICE), vs.to(DEVICE), points.to(DEVICE))


set_seed(0) # for default intersection
default_intersections = get_intersections()


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
def render_twolayer(layer0, layer1):
    WOs, IDISs, WIs, Us, Vs, POINTs = default_intersections

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

    layer0 = Layer(0.5, 0.5, [0.4, 0.2, 0.7], [1, 1, 1], True)
    layer0.set_map("orientation", spectrum2vector(readexr("textures/orientation.exr")), tiles=(80, 80))
    layer0.to(DEVICE)

    layer1 = Layer(0.3, 1.0, [0.4, 0.2, 0.7], [0.2], False)
    layer1.set_map("specular", sRGB2linear(readimg("textures/albedo2.png")))
    layer1.to(DEVICE)

    rendered = render_twolayer(layer0, layer1)
    # rendered = render_singlelayer(layer0)

    writeimg("rendered.png", rendered)




