import pygame
import random

_ENT_ID_COUNTER = 0
CELLSIZE = 64


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

    def get_rect(self, scale=1) -> pygame.Rect:
        return pygame.Rect(self.xy[0] * scale, self.xy[1] * scale, scale, scale)

    def update(self):
        pass

    def get_inset(self):
        return 8

    def get_line_width(self):
        return 4

    def draw_at(self, rect, surf):
        my_rect = self.get_rect(scale=CELLSIZE).inflate(-self.get_inset() * 2, -self.get_inset() * 2)
        pygame.draw.rect(surf, self.color, my_rect, width=self.get_line_width())

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)


class Forklift(Entity):

    def __init__(self, xy, direction=(0, -1)):
        super().__init__(xy, 0, 6, 100, "gold")
        self.direction = direction

    def rotate(self, steps_cw):
        # TODO can't rotate if there's not room
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        idx = dirs.index(self.direction)
        new_idx = (idx + steps_cw) % len(dirs)
        self.direction = dirs[new_idx]

    def move(self, steps):
        self.xy = (self.xy[0] + self.direction[0] * steps,
                   self.xy[1] + self.direction[1] * steps)

    def draw_at(self, rect, surf):
        inset = self.get_inset()
        color_darker = pygame.Color(self.color).lerp((0, 0, 0), 0.25)

        pygame.draw.line(surf, color_darker, (rect[0] + inset, rect[1] + inset),
                         (rect[0] + rect[2] - inset, rect[1] + rect[3] - inset), width=self.get_line_width())
        pygame.draw.line(surf, color_darker, (rect[0] + rect[2] - inset, rect[1] + inset),
                         (rect[0] + inset, rect[1] + rect[3] - inset), width=self.get_line_width())

        super().draw_at(rect, surf)

        rect.x += rect.width * self.direction[0]
        rect.y += rect.height * self.direction[1]
        rect.inflate_ip(-inset * 2, -inset * 2)

        # draw the fork
        fork_rect = rect.inflate(-8, -8)
        pygame.draw.rect(surf, self.color, fork_rect, width=self.get_line_width())


class Plank(Entity):
    pass


class Box(Entity):
    pass


class Level:
    def __init__(self):
        self.entities = []
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

    def draw_all(self, surf, camera_rect):
        cell_rect_size = (
            CELLSIZE * surf.get_width() / camera_rect[2],
            CELLSIZE * surf.get_height() / camera_rect[3]
        )
        for xy in self.terrain:
            cell_rect = pygame.Rect(xy[0] * cell_rect_size[0], xy[1] * cell_rect_size[1],
                                    cell_rect_size[0], cell_rect_size[1])
            z = self.terrain[xy]
            if z >= 0:
                base_color = pygame.Color("gray20")
                light_color = pygame.Color("gray40")
                color = base_color.lerp(light_color, min(z / 16.0, 1.0))
                pygame.draw.rect(surf, color, cell_rect.inflate(-2, -2), width=4)

        for ent in self.entities:
            ent_rect = ent.get_rect(scale=CELLSIZE)
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

    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Forklift Demo")
    camera = pygame.Rect(0, 0, screen.get_width(), screen.get_height())

    clock = pygame.time.Clock()
    dt = 0

    level = build_sample_level(8, 5)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_w:
                    level.get_forklift().move(1)
                elif e.key == pygame.K_s:
                    level.get_forklift().move(-1)
                elif e.key == pygame.K_a:
                    level.get_forklift().rotate(-1)
                elif e.key == pygame.K_d:
                    level.get_forklift().rotate(1)

        screen.fill("black")
        level.draw_all(screen, camera)

        pygame.display.flip()
        dt = clock.tick(60)