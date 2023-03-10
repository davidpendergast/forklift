import pygame
import random

from collections import deque

import src.utils.util as util
import src.utils.colorutils as colorutils


def save_to_png(surface: pygame.Surface, filepath):
    pygame.image.save(surface, filepath)


def bft(start_pos, cond, with_diags=False, bound_rect=None, rng_seed=None):
    """breadth-first traverse"""
    seen = set()
    q = deque()
    seen.add(start_pos)
    q.append(start_pos)

    if rng_seed is not None:
        random.seed(rng_seed)

    while len(q) > 0:
        n = q.pop()
        if cond(n):
            yield n

            neighbors = [m for m in util.neighbors(n[0], n[1], and_diags=with_diags)]
            if rng_seed is not None:
                random.shuffle(neighbors)

            for m in neighbors:
                if m not in seen and (bound_rect is None or util.rect_contains(bound_rect, m)):
                    q.append(m)
                seen.add(m)


def rm_alpha(color):
    if len(color) == 4:
        return (color[0], color[1], color[2])
    else:
        return color


def add_alpha(color, val=255):
    if len(color) == 3:
        return (color[0], color[1], color[2], val)
    else:
        return color


def flood_fill(surface: pygame.Surface, target_colors, start_pos, bound_rect=None, rng_seed=None):
    bound_rect = _make_bound_for_surface(surface, bound_rect)

    def _cond(xy):
        color = rm_alpha(tuple(surface.get_at(xy)))
        return color in target_colors

    return [p for p in bft(start_pos, _cond, bound_rect=bound_rect, rng_seed=rng_seed)]


def find_color_regions(surface: pygame.Surface, bound_rect=None, colors_to_include=None):
    bound_rect = _make_bound_for_surface(surface, bound_rect)
    res = {}  # color -> list of sets of (x, y)

    all_points = set()
    for x in range(bound_rect[0], bound_rect[0] + bound_rect[2]):
        for y in range(bound_rect[1], bound_rect[1] + bound_rect[3]):
            all_points.add((x, y))

    if colors_to_include is not None:
        for c in colors_to_include:
            res[c] = []

    while len(all_points) > 0:
        pt = all_points.pop()
        pt_color = rm_alpha(tuple(surface.get_at(pt)))

        if colors_to_include is not None and pt_color not in colors_to_include:
            continue
        if pt_color not in res:
            res[pt_color] = []
        region = set()
        for r in flood_fill(surface, [pt_color], pt, bound_rect=bound_rect):
            region.add(r)
            if r in all_points:
                all_points.remove(r)
        res[pt_color].append(region)

    return res


def _make_bound_for_surface(surface: pygame.Surface, bound_rect):
    max_rect = [0, 0, surface.get_width(), surface.get_height()]
    if bound_rect is None:
        return max_rect
    else:
        return util.get_rect_intersect(bound_rect, max_rect)


def region_color_fill(surface: pygame.Surface, regions, color_provider):
    """
    :param color_provider: lambda region -> color
    """
    for reg in regions:
        color = color_provider(reg)
        for r in reg:
            surface.set_at(r, add_alpha(color))


def find_bounding_rect(search_rect, sheet, keep_horz=False, keep_vert=False):
    if keep_horz and keep_vert:
        return search_rect

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    if keep_horz:
        min_x = search_rect[0]
        max_x = search_rect[0] + search_rect[2] - 1

    if keep_vert:
        min_y = search_rect[1]
        max_y = search_rect[1] + search_rect[3] - 1

    sheet_size = sheet.get_size()
    sheet.lock()
    for x in range(search_rect[0], search_rect[0] + search_rect[2]):
        for y in range(search_rect[1], search_rect[1] + search_rect[3]):
            if 0 <= x < sheet_size[0] and 0 <= y < sheet_size[1]:
                color = sheet.get_at((x, y))
                if not colorutils.is_transparent(color):
                    if min_x is None:
                        min_x = x
                        max_x = x
                    else:
                        min_x = min(x, min_x)
                        max_x = max(x, max_x)

                    if min_y is None:
                        min_y = y
                        max_y = y
                    else:
                        min_y = min(y, min_y)
                        max_y = max(y, max_y)
    sheet.unlock()

    if min_x is None and min_y is None:
        return [search_rect[0], search_rect[1], 0, 0]
    elif min_x is None:
        return [search_rect[0], min_y, 0, max_y - min_y + 1]
    elif min_y is None:
        return [min_x, search_rect[1], max_x - min_x + 1, 0]
    else:
        return [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]


def draw_decay_animation_effect(src_sheet, src_rect, n_frames, dest_sheet, dest_rect_provider,
                                full_decay_rect_provider, partial_decay_rect_provider,
                                decay_chance_provider=lambda i, xy: 0.05):
    """
    src_sheet: Surface containing the source image
    src_rect: Location of the source image
    n_frames: Number of frames to draw
    dest_sheet: Surface to draw the decayed images
    dest_rect_provider: frm_idx -> rect
    full_decay_rect_provider: frm_idx -> rect
    partial_decay_rect_provider: frm_idx -> rect
    decay_chance_provider: frm_idx, xy -> rect

    returns: list of dest rects drawn
    """
    res = []
    decayed = set()  # set of decayed pixels
    for i in range(0, n_frames):
        dest_rect = dest_rect_provider(i)
        full_decay_rect = full_decay_rect_provider(i)
        partial_decay_rect = partial_decay_rect_provider(i)

        for x in range(0, min(dest_rect[2], src_rect[2])):
            for y in range(0, min(dest_rect[3], src_rect[3])):
                src_xy = (src_rect[0] + x, src_rect[1] + y)
                if src_xy in decayed:
                    continue
                elif util.rect_contains(full_decay_rect, src_xy):
                    decayed.add(src_xy)
                    continue
                elif util.rect_contains(partial_decay_rect, src_xy):
                    decay_chance = decay_chance_provider(i, src_xy)
                    if random.random() < decay_chance:
                        decayed.add(src_xy)
                        continue

                color = src_sheet.get_at(src_xy)
                dest_xy = (dest_rect[0] + x, dest_rect[1] + y)
                dest_sheet.set_at(dest_xy, color)

        res.append(dest_rect)
    return res


def draw_vertical_line_phasing_animation(src_sheet, src_rect, n_frames, dest_sheet, dest_pos_provider,
                                         fade_out=True, rand_seed=None, min_fade_dur=0):
    if rand_seed is not None:
        random.seed(rand_seed)
    res = []

    start_and_end_times = []
    for i in range(0, src_rect[2]):
        start_t = random.randint(0, n_frames - min_fade_dur - 1)
        end_t = random.randint(start_t + min_fade_dur, n_frames - 1)
        start_and_end_times.append((start_t, end_t))

    for i in range(0, n_frames):
        dest_xy = dest_pos_provider(i)
        for x in range(0, src_rect[2]):
            start_t, end_t = start_and_end_times[x]
            if i <= start_t:
                pcnt_col_to_draw = 1
                fade_factor = 1
            elif i > end_t:
                pcnt_col_to_draw = 0
                fade_factor = 0
            else:
                pcnt_col_to_draw = 1 - (i - start_t) / (end_t - start_t)
                bifurcation = 0.75
                if pcnt_col_to_draw >= bifurcation:
                    fade_factor = 1
                else:
                    fade_factor = pcnt_col_to_draw / bifurcation

            if not fade_out:
                fade_factor = 1 - fade_factor
                pcnt_col_to_draw = 1 - pcnt_col_to_draw

            for y in range(0, src_rect[3]):
                if (fade_out and y / src_rect[3] >= 1 - pcnt_col_to_draw) or (not fade_out and y / src_rect[3] < pcnt_col_to_draw):
                    px_val = add_alpha(src_sheet.get_at((x + src_rect[0], y + src_rect[1])))
                    px_val = (px_val[0], px_val[1], px_val[2], int(px_val[3] * fade_factor))
                    dest_sheet.set_at((x + dest_xy[0], y + dest_xy[1]), px_val)

            res.append([dest_xy[0], dest_xy[1], src_rect[2], src_rect[3]])

    return res


def draw_rotated_sprite(src_sheet, src_rect, dest_sheet, dest_rect, rot):
    temp_surf = pygame.Surface((src_rect[2], src_rect[3]), pygame.SRCALPHA, 32)
    temp_surf.blit(src_sheet, (0, 0), area=src_rect)
    rotated_surf = pygame.transform.rotate(temp_surf, rot * 360)

    temp_surf_2 = pygame.Surface((dest_rect[2], dest_rect[3]), pygame.SRCALPHA, 32)
    temp_surf_2.blit(rotated_surf, (dest_rect[2] // 2 - rotated_surf.get_width() // 2,
                                    dest_rect[3] // 2 - rotated_surf.get_height() // 2))

    dest_sheet.blit(temp_surf_2, dest_rect)


def draw_with_transparency(src_sheet, src_rect, dest_sheet, dest_pos, alpha):
    """
    :param alpha: a value from [0.0, 1.0] where 0.0 is fully transparent
    """
    draw_wtih_color_xform(src_sheet, src_rect, dest_sheet, dest_pos,
                          lambda color: (color[0], color[1], color[2], util.bound(color[3] * alpha, 0, 1)))


def draw_wtih_color_xform(src_sheet, src_rect, dest_sheet, dest_xy, xform=lambda rgba: rgba):
    for x in range(src_rect[0], src_rect[0] + src_rect[2]):
        for y in range(src_rect[1], src_rect[1] + src_rect[3]):
            orig_rgba = add_alpha(src_sheet.get_at((x, y)))
            orig_rgba = colorutils.to_float(orig_rgba)

            new_rgba = xform(orig_rgba)
            new_rgba = colorutils.to_int(new_rgba)
            dest_pos = (dest_xy[0] + x - src_rect[0], dest_xy[1] + y - src_rect[1])
            dest_sheet.set_at(dest_pos, new_rgba)


def apply_darkness(src_sheet, src_rect, dest_sheet, dest_xy, darkness, contrast_preserving=True, max_change=-1):
    """
    :param darkness: a value from [0.0, 1.0) where 1.0 is maximum darkness
    :param contrast_preserving: whether to use Ghast's special sauce
    :param max_change: the max amount to shift a color (out of 255)
    """
    dest_rect = [dest_xy[0], dest_xy[1], src_rect[2], src_rect[3]]
    dest_sheet.blit(src_sheet, dest_rect, area=src_rect)

    for x in range(dest_rect[0], dest_rect[0] + dest_rect[2]):
        for y in range(dest_rect[1], dest_rect[1] + dest_rect[3]):
            rgb = list(dest_sheet.get_at((x, y)))
            for i in range(0, 3):
                val = rgb[i] / 255
                if contrast_preserving:
                    # blend the value with a crazy curve
                    new_val = (1 - darkness) * val ** (1 / (1 - darkness))
                else:
                    # just add black with alpha equal to darkness.
                    # in other words: alpha * 0 + (1 - alpha) * val
                    new_val = (1 - darkness) * val

                if max_change >= 0:
                    new_val = max(val - max_change, new_val)

                rgb[i] = int(255 * new_val)
            dest_sheet.set_at((x, y), rgb)


def rainbowfill(surface, rect=None, rainbow_height=None):
    if rect is None:
        rect = [0, 0, surface.get_width(), surface.get_height()]
    for x in range(rect[0], rect[0] + rect[2]):
        for y in range(rect[1], rect[1] + rect[3]):
            if rainbow_height is None:
                pcnt_y = (y - rect[1]) / rect[3]
            else:
                pcnt_y = ((y - rect[1]) % rainbow_height) / rainbow_height

            h = pcnt_y * 360
            rgb = colorutils.hsv_to_rgb(h, 1, 1)
            rgb_int = colorutils.to_int(rgb)
            surface.set_at((x, y), rgb_int)


