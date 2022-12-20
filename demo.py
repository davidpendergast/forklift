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


def _next_ent_id():
    global _ENT_ID_COUNTER
    _ENT_ID_COUNTER += 1
    return _ENT_ID_COUNTER - 1


class Entity:

    def __init__(self, xy, z, height, weight, color, liftable=False):
        self.uid = _next_ent_id()
        self.height = height
        self.liftable = liftable
        self.weight = weight
        self.color = color

        self.xy = xy
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
        return pygame.Rect(self.xy[0] * scale, self.xy[1] * scale, scale, scale)

    def get_z(self, xy=None):
        return self.z

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
        pygame.draw.rect(surf, self.color, my_rect, width=self.get_line_width())

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)


class Forklift(Entity):

    def __init__(self, xy, direction=(0, -1)):
        super().__init__(xy, 0, 6, 100, "gold")
        self.direction = direction
        self.lift_rel_z = 0

    def get_lift_z(self):
        return self.z + self.lift_rel_z

    def get_fork_xy(self):
        return (self.xy[0] + self.direction[0],
                self.xy[1] + self.direction[1])

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
            new_lift_xy = self.xy[0] + new_dir[0], self.xy[1] + new_dir[1]
            if not world.is_empty(new_lift_xy, z=lift_z, ignoring=(self,)):
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
        dest_xy = (self.xy[0] + steps * self.direction[0], self.xy[1] + steps * self.direction[1])
        if world is None:
            return dest_xy
        start_z = self.z

        neg = -1 if steps < 0 else 1
        steps = abs(steps)
        for i in range(1, steps+1):
            xy = (self.xy[0] + i * self.direction[0] * neg, self.xy[1] + i * self.direction[1] * neg)
            fork_xy = (xy[0] + self.direction[0], xy[1] + self.direction[1])

            world_z = world.get_z(xy, ignoring=(self,))
            if world_z == -1 or abs(world_z - start_z) > Z_STEP_THRESH:
                return None  # cab is blocked
            elif not world.is_empty(fork_xy, z=world_z + self.lift_rel_z, ignoring=(self,)):
                return None  # fork is blocked

        return dest_xy

    def move(self, steps) -> bool:
        xy = self.can_move(steps)
        if xy is not None:
            self.z = self.get_world().get_z(xy, ignoring=(self,))
            self.xy = xy
            return True
        return False

    def can_raise_fork(self, steps) -> int:
        fork_xy = self.get_fork_xy()

        neg = -1 if steps < 0 else 1
        steps = abs(steps)
        for i in range(1, steps+1):
            rel_z = self.lift_rel_z + i * neg
            if not (0 <= rel_z <= FORK_MAX_HEIGHT):
                return None
            elif not self.get_world().is_empty(fork_xy, z=self.z + rel_z, ignoring=(self,)):
                return None

        return self.lift_rel_z + steps * neg

    def raise_fork(self, steps) -> bool:
        new_rel_z = self.can_raise_fork(steps)
        if new_rel_z is not None:
            self.lift_rel_z = new_rel_z
            return True
        return False

    def draw_at(self, rect, surf):
        inset = self.get_inset()
        color = self.get_color(self.z)

        pygame.draw.line(surf, color, (rect[0] + inset, rect[1] + inset),
                         (rect[0] + rect[2] - inset, rect[1] + rect[3] - inset), width=self.get_line_width())
        pygame.draw.line(surf, color, (rect[0] + rect[2] - inset, rect[1] + inset),
                         (rect[0] + inset, rect[1] + rect[3] - inset), width=self.get_line_width())

        super().draw_at(rect, surf)

        rect.x += rect.width * self.direction[0]
        rect.y += rect.height * self.direction[1]
        rect.inflate_ip(-inset * 2, -inset * 2)

        # draw the fork
        fork_rect = rect.inflate(-8, -8)
        fork_color = self.get_world().lerp_color_for_z(self.color, self.get_lift_z())
        pygame.draw.rect(surf, fork_color, fork_rect, width=self.get_line_width())


class Plank(Entity):
    pass


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

    def get_z(self, xy, ignoring=()):
        # TODO handle caves
        max_z = self.get_terrain_z(xy)
        for e in self.all_entities_at(xy, ignoring=ignoring):
            max_z = max(max_z, e.get_z(xy) + e.get_height(xy))
        return max_z

    def is_empty(self, xy, z=0, ignoring=()):
        level_z = self.get_z(xy, ignoring=ignoring)
        if isinstance(z, (int, float)):
            return level_z <= z
        else:
            for z_ in z:
                if z < level_z:
                    return False
            return True

    def all_entities_at(self, xy, ignoring=()):
        for e in self.entities:
            if e not in ignoring and e.get_rect().collidepoint(*xy):
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
            CELL_SIZE * surf.get_width() / camera_rect[2],
            CELL_SIZE * surf.get_height() / camera_rect[3]
        )
        for xy in self.terrain:
            cell_rect = pygame.Rect(xy[0] * cell_rect_size[0], xy[1] * cell_rect_size[1],
                                    cell_rect_size[0], cell_rect_size[1])
            cell_rect = self.world_rect_to_screen(cell_rect, surf.get_size(), camera_rect)
            z = self.terrain[xy]
            if z >= 0:
                base_color = pygame.Color("gray20")
                color = self.lerp_color_for_z(base_color, z)
                pygame.draw.rect(surf, color, cell_rect.inflate(-2, -2), width=4)

        for ent in self.entities:
            ent_rect = ent.get_rect(scale=CELL_SIZE)
            ent_screen_rect = self.world_rect_to_screen(ent_rect, surf.get_size(), camera_rect)
            ent.draw_at(ent_screen_rect, surf)


def build_sample_level(w, h) -> Level:
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

    forklift = Forklift(f_xy)
    res.add(forklift)

    return res


if __name__ == "__main__":
    pygame.init()

    level = build_sample_level(*LEVEL_DIMS)

    pygame.display.set_caption("Forklift Demo")
    screen = pygame.display.set_mode((640, 480))

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
                    level.get_forklift().raise_fork(1)
                elif e.key in (pygame.K_c,):
                    level.get_forklift().raise_fork(-1)

        screen.fill("black")
        level.draw_all(screen, camera)

        pygame.display.flip()
        dt = clock.tick(60)