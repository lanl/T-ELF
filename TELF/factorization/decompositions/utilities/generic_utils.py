import warnings
from matplotlib.transforms import Affine2D
from matplotlib.spines import Spine
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.path import Path
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
from itertools import product, repeat
import psutil
import numpy as np

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse.linalg
    import cupyx.scipy.linalg
    
    HAS_CUPY = True
except:
    HAS_CUPY = False
    
import scipy
import scipy.sparse
import scipy.sparse.linalg
import pathos

mp = pathos.helpers.mp


def grid_eval(fn, grid, num_cpus=None, **kwargs):
    def kwargs_starmap_wrap(fn, args, kwargs):
        return fn(*args, **kwargs)

    arguments_grid = list(product(*grid))
    kwargs_starmap_args = zip(repeat(fn), arguments_grid, repeat(kwargs))
    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False)
    with mp.get_context("spawn").Pool(num_cpus) as pool:
        out = pool.starmap(kwargs_starmap_wrap, kwargs_starmap_args)
        pool.close()
        pool.join()

    if isinstance(out[0], (tuple, list)):
        np = get_np(*out[0], use_gpu=False)
        out = list(map(list, zip(*out)))
        out = [np.array(o) for o in out]
        out = [o.reshape([len(g) for g in grid] + list(o.shape[1:])) for o in out]
        out = [
            o.transpose(
                [i for i in range(len(grid), o.ndim)] + [i for i in range(len(grid))]
            )
            for o in out
        ]
    else:
        np = get_np(out[0], use_gpu=False)
        out = np.array(out)
        out = out.reshape([len(g) for g in grid] + list(out.shape[1:]))
        out = out.transpose(
            [i for i in range(len(grid), out.ndim)] + [i for i in range(len(grid))]
        )
    return out


def get_np(*args, **kwargs):
    if "use_gpu" in kwargs:
        use_gpu = kwargs["use_gpu"]
        del kwargs["use_gpu"]
    else:
        use_gpu = False
    if use_gpu and HAS_CUPY:
        try:
            xp = cp.get_array_module(*args, **kwargs)
        except Exception as e:
            warnings.warn(f'Attempted to use Cupy but got the error: {e}')
            xp = np
    else:
        xp = np

    return xp

def get_cupyx(use_gpu):
    if use_gpu and HAS_CUPY:
        return cupyx, HAS_CUPY
    else:
        return scipy, HAS_CUPY

def get_scipy(*args, **kwargs):
    if "use_gpu" in kwargs:
        use_gpu = kwargs["use_gpu"]
        del kwargs["use_gpu"]
    else:
        use_gpu = False

    if use_gpu and HAS_CUPY:
        try:
            xpy = cupyx.scipy.get_array_module(*args, **kwargs)
        except Exception as e:
            warnings.warn(f'Attempted to use xCupy but got the error: {e}')
            xpy = scipy
    else:
        xpy = scipy

    return xpy


def update_opts(defaults, custom):
    if custom is None:
        custom = {}
    for k, v in custom.items():
        if k not in defaults:
            warnings.warn("key not found in defaults dictionary, key: " + str(k))
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(custom.get(k, {}), dict):
            custom[k] = update_opts(v, custom.get(k, {}))
        else:
            custom[k] = custom.get(k, v)
    return custom


def bary_proj(num_vars):
    """
    Computes generalized Barycentric coordinates from an activation matrix.

    Args:
        num_vars (int): Number of extreme rays in barycentric plot.

    Returns:
        theta (ndarray): Angles of extreme points.
    """

    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    proj_name = "bary{:d}".format(num_vars)

    class BaryAxes(PolarAxes):
        name = proj_name
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")
            self.set_varlabels(list(range(num_vars)))
            self.set_yticks([])
            self.grid(b=None)
            self.set_ylim([0, 1])

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")

        def _gen_axes_spines(self):
            # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
            spine = Spine(
                axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars)
            )
            # unit_regular_polygon gives a polygon of radius 1 centered at
            # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
            # 0.5) in axes coordinates.
            spine.set_transform(
                Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
            )
            return {"polar": spine}

    register_projection(BaryAxes)
    return proj_name
