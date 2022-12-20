import pygame
import random
import math
import typing

_ENT_ID_COUNTER = 0

LEVEL_DIMS = 8, 5
CELL_SIZE = 64
CELL_HEIGHT = 8
Z_STEP_THRESH = 2
FORK_MAX_HEIGHT = 8

FONTS: typing.Dict[int, pygame.font.Font] = {}
CACHED_TEXT = {}


def get_text_surf(text, size, color):
    if size not in FONTS:
        FONTS[size] = pygame.font.Font("assets/small_pixel.ttf", size)
    color = pygame.Color(color)
    text_key = (text, size, color.r, color.g, color.b)
    if text_key not in CACHED_TEXT:
        CACHED_TEXT[text_key] = FONTS[size].render(text, False, color, (0, 0, 0, 0))
    return CACHED_TEXT[text_key]


def draw_text(surf: pygame.Surface, xy, text, size=16, color="white"):
    surf.blit(get_text_surf(text, size, color), xy)


def _next_ent_id():
    global _ENT_ID_COUNTER
    _ENT_ID_COUNTER += 1
    return _ENT_ID_COUNTER - 1


class Entity:

    def __init__(self, rect, z, height, weight, color, liftable=False):
        self.uid = _next_ent_id()
        self.height = height
        self.liftable = liftable
        self.weight = weight
        self.color = color

        self.rect = rect
        self.z = z

        self._world = None  # set by Level

    def get_color(self, z_offset):
        if z_offset <= 0:
            return self.color
        else:
            # TODO hmmm
            a = max(0., min(1.0, math.atan(8 * z_offset / CELL_HEIGHT) / (math.pi / 2)))
            return pygame.Color(self.color).lerp((255, 255, 255), a)

    def get_rect(self, scale=1) -> pygame.Rect:
        return pygame.Rect(self.rect.x * scale, self.rect.y * scale,
                           self.rect.width * scale, self.rect.height * scale)

    def xy(self):
        return self.get_rect().topleft

    def get_z(self, xy=None):
        return self.z

    def is_liftable(self):
        return self.liftable

    def get_height(self, xy=None):
        return self.height

    def get_world(self) -> "Level":
        return self._world

    def update(self):
        pass

    def get_inset(self):
        return 8

    def get_line_width(self):
        return 4

    def draw_at(self, rect, surf):
        my_rect = rect.inflate(-self.get_inset() * 2, -self.get_inset() * 2)
        pygame.draw.rect(surf, self.get_color(self.get_z()), my_rect, width=self.get_line_width())

        z_text_pos = my_rect[0] + self.get_inset(), my_rect[1] + self.get_inset()
        z_text = str(self.get_z())
        draw_text(surf, z_text_pos, z_text, color=self.get_color(self.get_z()))

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)


def add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])


class Forklift(Entity):

    def __init__(self, xy, z, direction=(0, -1)):
        super().__init__(pygame.Rect(xy[0], xy[1], 1, 1), z, 6, 100, "gold")
        self.direction = direction
        self.lift_rel_z = 0

    def get_lift_z(self):
        return self.z + self.lift_rel_z

    def get_fork_xy(self):
        return (self.xy()[0] + self.direction[0],
                self.xy()[1] + self.direction[1])

    def can_rotate(self, steps_cw) -> typing.Tuple[int, int]:
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        idx = dirs.index(self.direction)
        res_dir = dirs[(idx + steps_cw) % len(dirs)]

        world = self.get_world()
        if world is None:
            return res_dir

        step_size = 1
        if steps_cw < 0:
            step_size = 3
            steps_cw = abs(steps_cw)

        lift_z = self.z + self.lift_rel_z

        for i in range(1, steps_cw+1):
            new_dir = dirs[(idx + step_size * i) % len(dirs)]
            new_lift_xy = self.xy()[0] + new_dir[0], self.xy()[1] + new_dir[1]
            if not world.is_empty(new_lift_xy, z=lift_z, ignoring=(self,), ignore_cond=lambda e: e.is_liftable()):
                return None

        return res_dir

    def rotate(self, steps_cw) -> bool:
        new_dir = self.can_rotate(steps_cw)
        if new_dir is not None:
            self.direction = new_dir
            return True
        return False

    def can_move(self, steps) -> typing.Tuple[int, int]:
        world = self.get_world()
        dest_xy = (self.xy()[0] + steps * self.direction[0], self.xy()[1] + steps * self.direction[1])
        if world is None:
            return dest_xy
        start_z = self.z

        neg = -1 if steps < 0 else 1
        steps = abs(steps)
        for i in range(1, steps+1):
            xy = (self.xy()[0] + i * self.direction[0] * neg, self.xy()[1] + i * self.direction[1] * neg)
            fork_xy = (xy[0] + self.direction[0], xy[1] + self.direction[1])

            world_z = world.get_z(xy, ignoring=(self,))
            if world_z == -1 or abs(world_z - start_z) > Z_STEP_THRESH:
                return None  # cab is blocked
            elif not world.is_empty(fork_xy, z=world_z + self.lift_rel_z, ignoring=(self,), ignore_cond=lambda e: e.is_liftable()):
                return None  # fork is blocked

        return dest_xy

    def move(self, steps) -> bool:
        xy = self.can_move(steps)
        if xy is not None:
            self.z = self.get_world().get_z(xy, ignoring=(self,))
            self.rect.topleft = xy
            return True
        return False

    def can_raise_fork(self, steps, or_less=True) -> int:
        fork_xy = self.get_fork_xy()

        neg = -1 if steps < 0 else 1
        steps = abs(steps)
        best_rel_z = None
        for i in range(1, steps+1):
            rel_z = self.lift_rel_z + i * neg
            if not (0 <= rel_z <= FORK_MAX_HEIGHT) \
                    or not self.get_world().is_empty(fork_xy, z=self.z + rel_z, ignoring=(self,)):
                return best_rel_z if or_less else None
            else:
                best_rel_z = rel_z
        max_rel_z = self.lift_rel_z + steps * neg
        if or_less:
            return best_rel_z
        else:
            return max_rel_z

    def raise_fork(self, steps, or_less=True) -> bool:
        new_rel_z = self.can_raise_fork(steps, or_less=or_less)
        if new_rel_z is not None:
            self.lift_rel_z = new_rel_z
            return True
        return False

    def draw_at(self, rect, surf):
        inset = self.get_inset()
        color = pygame.Color(self.get_color(self.z))
        color2 = color.lerp("black", 0.15)

        pygame.draw.line(surf, color2,
                         (rect[0] + inset*2, rect[1] + inset*2),
                         (rect[0] + rect[2] - inset*2, rect[1] + rect[3] - inset*2), width=3)
        pygame.draw.line(surf, color2,
                         (rect[0] + rect[2] - inset*2, rect[1] + inset*2),
                         (rect[0] + inset*2, rect[1] + rect[3] - inset*2), width=3)

        super().draw_at(rect, surf)
        fork_z = self.get_lift_z()
        fork_color = self.get_world().lerp_color_for_z(color2, fork_z)

        text_sz = 16
        fork_z_text_pos = (rect.x + 2 * self.get_inset(), rect.bottom - text_sz - 2 * self.get_inset())
        draw_text(surf, fork_z_text_pos, str(fork_z), size=text_sz, color=fork_color)

        rect.x += rect.width * self.direction[0]
        rect.y += rect.height * self.direction[1]
        rect.inflate_ip(-inset * 2, -inset * 2)

        # draw the fork
        fork_rect = rect.inflate(-8, -8)
        fork_rect.x -= self.direction[0] * inset*2
        fork_rect.y -= self.direction[1] * inset*2

        if self.direction[0] == 0:
            fork_rect.inflate_ip(-inset * 2, 0)
        else:
            fork_rect.inflate_ip(0, -inset * 2)

        pygame.draw.rect(surf, fork_color, fork_rect, width=self.get_line_width())


class Plank(Entity):

    def __init__(self, rect, z, height=1):
        super().__init__(rect, z, height, rect.width * rect.height * height, "tan", liftable=True)


class Box(Entity):
    pass


class Level:
    def __init__(self):
        self.entities: typing.List[Entity] = []
        self.terrain = {}

    def set_terrain(self, xy, height=0):
        self.terrain[xy] = height

    def add(self, ent):
        ent._world = self
        self.entities.append(ent)

    def remove(self, ent):
        self.entities.remove(ent)
        ent._world = None

    def get_forklift(self):
        for e in self.entities:
            if isinstance(e, Forklift):
                return e
        return None

    def get_terrain_z(self, xy):
        if xy not in self.terrain:
            return -1
        else:
            return self.terrain[xy]

    def get_z(self, xy, ignoring=(), ignore_cond=None):
        # TODO handle caves
        max_z = self.get_terrain_z(xy)
        for e in self.all_entities_at(xy, ignoring=ignoring, ignore_cond=ignore_cond):
            max_z = max(max_z, e.get_z(xy) + e.get_height(xy))
        return max_z

    def is_empty(self, xy, z=0, ignoring=(), ignore_cond=None):
        level_z = self.get_z(xy, ignoring=ignoring, ignore_cond=ignore_cond)
        if isinstance(z, (int, float)):
            return level_z <= z
        else:
            for z_ in z:
                if z < level_z:
                    return False
            return True

    def all_entities_at(self, xy, ignoring=(), ignore_cond=None):
        for e in self.entities:
            if e not in ignoring and e.get_rect().collidepoint(*xy) \
                    and (ignore_cond is None or not ignore_cond(e)):
                yield e

    def world_xy_to_screen(self, xy, screen_size, camera_rect):
        c_xy = (xy[0] - camera_rect[0], xy[1] - camera_rect[1])
        return (c_xy[0] * screen_size[0] / camera_rect[2],
                c_xy[1] * screen_size[1] / camera_rect[3])

    def screen_xy_to_world(self, s_xy, screen_size, camera_rect):
        c_xy = (s_xy[0] * camera_rect[2] / screen_size[0],
                s_xy[1] * camera_rect[3] / screen_size[1])
        return (c_xy[0] + camera_rect[0], c_xy[1] + camera_rect[1])

    def world_rect_to_screen(self, rect, screen_size, camera_rect):
        p1 = rect[0], rect[1]
        p2 = rect[0] + rect[2], rect[1] + rect[3]
        s_p1 = self.world_xy_to_screen(p1, screen_size, camera_rect)
        s_p2 = self.world_xy_to_screen(p2, screen_size, camera_rect)
        return pygame.Rect((s_p1[0], s_p1[1], s_p2[0] - s_p1[0], s_p2[1] - s_p1[1]))

    def screen_rect_to_world(self, s_rect, screen_size, camera_rect):
        s_p1 = s_rect[0], s_rect[1]
        s_p2 = s_rect[0] + s_rect[2], s_rect[1] + s_rect[3]
        p1 = self.screen_xy_to_world(s_p1, screen_size, camera_rect)
        p2 = self.screen_xy_to_world(s_p2, screen_size, camera_rect)
        return pygame.Rect((p1[0], p1[1], p2[0] - p1[0], s_p2[1] - s_p1[1]))

    def lerp_color_for_z(self, base_color, z, max_a=1.0, high_color="white"):
        a = max(0., min(max_a, math.atan(z / CELL_HEIGHT) / (math.pi / 2)))
        return pygame.Color(base_color).lerp(high_color, a)

    def draw_all(self, surf, camera_rect):
        cell_rect_size = (
            CELL_SIZE,
            CELL_SIZE
        )
        for xy in self.terrain:
            cell_rect = pygame.Rect(xy[0] * cell_rect_size[0],
                                    xy[1] * cell_rect_size[1],
                                    cell_rect_size[0], cell_rect_size[1])
            cell_rect = self.world_rect_to_screen(cell_rect, surf.get_size(), camera_rect)

            z = self.terrain[xy]
            if z >= 0:
                base_color = pygame.Color("gray20")
                color = self.lerp_color_for_z(base_color, z)
                pygame.draw.rect(surf, color, cell_rect.inflate(-2, -2), width=4)

                z_text_pos = cell_rect.right - 28, cell_rect.top + 16
                draw_text(surf, z_text_pos, str(z), color=color)

        for ent in self.entities:
            ent_rect = ent.get_rect(scale=CELL_SIZE)
            ent_screen_rect = self.world_rect_to_screen(ent_rect, surf.get_size(), camera_rect)
            ent.draw_at(ent_screen_rect, surf)


def build_sample_level(w, h, n_planks=15) -> Level:
    res = Level()
    f_xy = w // 2, h // 2
    for x in range(w):
        for y in range(h):
            r = random.random()
            if r < 0.8 or (x, y) == f_xy:
                res.set_terrain((x, y), 0)
            elif r < 0.9:
                res.set_terrain((x, y), 8)
            else:
                res.set_terrain((x, y), -1)

    for _ in range(n_planks):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = 1 + 2 * random.randint(0, 2)
        horz = random.random() < 0.5

        max_z_left = -1
        max_z_right = -1
        for i in range(0, length):
            xy_left = (x - i * (1 if horz else 0), y - i * (0 if horz else 1))
            max_z_left = max(res.get_z(xy_left), max_z_left)
            xy_right = (x + i * (1 if horz else 0), y + i * (0 if horz else 1))
            max_z_right = max(res.get_z(xy_right), max_z_right)
        if max_z_left == max_z_right != -1:
            r = pygame.Rect(x - (length - 1) // 2 * (1 if horz else 0),
                            y - (length - 1) // 2 * (0 if horz else 1),
                            length if horz else 1,
                            1 if horz else length)
            res.add(Plank(r, max_z_left))

    res.add(Forklift(f_xy, res.get_z(f_xy)))

    return res


if __name__ == "__main__":
    pygame.init()

    level = build_sample_level(*LEVEL_DIMS)

    pygame.display.set_caption("Forklift Demo")
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)

    camera = pygame.Rect((LEVEL_DIMS[0] * CELL_SIZE - screen.get_width()) // 2,
                         (LEVEL_DIMS[1] * CELL_SIZE - screen.get_height()) // 2,
                         screen.get_width(), screen.get_height())

    clock = pygame.time.Clock()
    dt = 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_r,):
                    level = build_sample_level(*LEVEL_DIMS)
                elif e.key in (pygame.K_w, pygame.K_UP):
                    level.get_forklift().move(1)
                elif e.key in (pygame.K_s, pygame.K_DOWN):
                    level.get_forklift().move(-1)
                elif e.key in (pygame.K_a, pygame.K_LEFT):
                    level.get_forklift().rotate(-1)
                elif e.key in (pygame.K_d, pygame.K_RIGHT):
                    level.get_forklift().rotate(1)
                elif e.key in (pygame.K_SPACE,):
                    level.get_forklift().raise_fork(FORK_MAX_HEIGHT)
                elif e.key in (pygame.K_c,):
                    level.get_forklift().raise_fork(-FORK_MAX_HEIGHT)
            elif e.type == pygame.VIDEORESIZE:
                camera = pygame.Rect((LEVEL_DIMS[0] * CELL_SIZE - screen.get_width()) // 2,
                                     (LEVEL_DIMS[1] * CELL_SIZE - screen.get_height()) // 2,
                                     screen.get_width(), screen.get_height())

        screen.fill("black")
        level.draw_all(screen, camera)

        pygame.display.flip()
        dt = clock.tick(60)