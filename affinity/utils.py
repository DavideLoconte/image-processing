from functools import lru_cache
from math import floor, ceil


@lru_cache(maxsize=8)
def grid(grid_size, shape):
    padding = size_to_padding(grid_size, shape)
    w = padding[0] + padding[2] + shape[-1]
    h = padding[1] + padding[3] + shape[-2]
    assert(w % grid_size == 0)
    assert(h % grid_size == 0)
    return w//grid_size, h//grid_size

@lru_cache(maxsize=8)
def position(grid_size, shape, pos):
    padding = size_to_padding(grid_size, shape)
    x, y = pos[0] + padding[0], pos[1] + padding[1]
    c, r = x//grid_size, y//grid_size
    nx, ny = x / grid_size - c, y / grid_size - r
    return c, r, nx, ny

@lru_cache(maxsize=8)
def relposition(grid_size, cell, pos):
    x, y = pos
    c, r = cell
    nx = x / grid_size - c
    ny = y / grid_size - r
    return nx, ny

@lru_cache(maxsize=8)
def size_to_padding(grid_size, shape):
    return (
        floor((grid_size - shape[-1] % grid_size) / 2) if shape[-1] % grid_size != 0 else 0,
        floor((grid_size - shape[-2] % grid_size) / 2) if shape[-2] % grid_size != 0 else 0,
        ceil((grid_size - shape[-1] % grid_size) / 2) if shape[-1] % grid_size != 0 else 0,
        ceil((grid_size - shape[-2] % grid_size) / 2) if shape[-2] % grid_size != 0 else 0
    )
