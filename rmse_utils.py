import numpy as np
import pystrum.pynd.ndutils as nd
import torch
import copy
import itertools


def prod_n(lst):
    prod = copy.deepcopy(lst[0])
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs, **kwargs):
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = copy.deepcopy(subs[-1])
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def interpn(vol, loc, interp_method='linear'):
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, -1)

    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)

    loc = loc.type(torch.FloatTensor)

    if isinstance(vol.shape, (torch.Size,)):
        volshape = list(vol.shape)
    else:
        volshape = vol.shape

    if interp_method == "linear":
        loc0 = torch.floor(loc)

        max_loc = [d - 1 for d in list(vol.shape)]

        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.type(torch.IntTensor) for f in loc0lst], [f.type(torch.IntTensor) for f in loc1]]

        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]

        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            idx = sub2ind(vol.shape[:-1], subs)
            idx = torch.as_tensor(idx, dtype=torch.long)
            vol_val = torch.reshape(vol, (-1, volshape[-1]))[idx]

            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)

            wt = torch.unsqueeze(wt, -1).cuda()

            interp_vol += wt * vol_val

    else:
        assert interp_method == "nearest"
        loc = torch.round(loc)
        roundloc = loc.type(torch.IntTensor)

        max_loc = [(d - 1).type(torch.IntTensor) for d in vol.shape]
        roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = torch.reshape(vol, (-1, vol.shape[-1]))[idx]

    return interp_vol


def point_spatial_transformer(x, sdt_vol_resize=1):

    surface_points, trf = x
    trf = trf * sdt_vol_resize
    surface_pts_D = surface_points.shape[-1]
    trf_D = trf.shape[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = torch.unsqueeze(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]

    fn = lambda x: interpn(x[0], x[1])

    diff = fn([trf, surface_points])
    # diff = x.map_(x, fn)
    ret = surface_points + diff

    if surface_pts_D == trf_D + 1:
        ret = torch.cat((ret, li_surface_pts), -1)
    return ret
