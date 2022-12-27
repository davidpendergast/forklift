import typing

import src.utils.util as util

_ENT_ID = 0


def _next_ent_id():
    global _ENT_ID
    _ENT_ID += 1
    return _ENT_ID - 1


class Entity:

    def __init__(self, xyz, cells, uid=None):
        self.uid = uid if uid is not None else _next_ent_id()
        self.xyz = xyz
        self.cells = set(cells)
        self._box = None
        self._normalize()

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def get_cells(self, absolute=True) -> typing.Set[typing.Tuple[int, int, int]]:
        x, y, z = self.xyz if absolute else (0, 0, 0)
        return set((c[0] + x, c[1] + y, c[2] + z) for c in self.cells)

    def get_mass(self) -> float:
        return len(self.cells)

    def get_center_of_mass(self, absolute=True) -> typing.Tuple[float, float, float]:
        sums = [0, 0, 0]
        mass = 0
        for x, y, z in self.get_cells(absolute=absolute):
            sums[0] += x + 0.5
            sums[1] += y + 0.5
            sums[2] += z + 0.5
            mass += 1
        return util.mult(sums, 1 / mass)

    def _calc_bounding_box(self, absolute=True):
        minx, miny, minz = float('inf'), float('inf'), float('inf')
        maxx, maxy, maxz = -float('inf'), -float('inf'), -float('inf')
        for (cx, cy, cz) in self.get_cells(absolute=absolute):
            minx = min(minx, cx)
            miny = min(miny, cy)
            minz = min(minz, cz)
            maxx = max(maxx, cx)
            maxy = max(maxy, cy)
            maxz = max(maxz, cz)

        if minx == float('inf'):
            return (*(self.xyz if absolute else (0, 0, 0)), 0, 0, 0)
        else:
            return (int(minx), int(miny), int(minz),
                    int(maxx - minx + 1),
                    int(maxy - miny + 1),
                    int(maxz - minz + 1))

    def get_bounding_box(self, absolute=True):
        if self._box is None:
            self._normalize()

        if absolute:
            return (self._box[0] + self.xyz[0],
                    self._box[1] + self.xyz[1],
                    self._box[2] + self.xyz[2],
                    self._box[3], self._box[4], self._box[5])
        else:
            return self._box

    def get_debug_color(self):
        raise (55, 55, 55)

    def copy(self, keep_uid=True) -> 'Entity':
        if type(self) is Entity:
            return Entity(self.xyz, self.cells, uid=self.uid if keep_uid else None)
        else:
            raise NotImplementedError

    def __contains__(self, xyz):
        xyz_int = (int(xyz[0]) - self.xyz[0],
                   int(xyz[1]) - self.xyz[1],
                   int(xyz[2]) - self.xyz[2])
        return xyz_int in self.cells

    def collides(self, other):
        cells1 = self.get_cells()
        cells2 = other.get_cells()
        return not cells1.isdisjoint(cells2)

    def collides_with_box(self, box):
        xy_rect = (box[0], box[1], box[3], box[4])
        xz_rect = (box[0], box[2], box[3], box[5])
        for c in self.get_cells():
            if (util.rect_contains(xy_rect, (c[0], c[1])) and
                    util.rect_contains(xz_rect, (c[0], c[2]))):
                return True
        return False

    def box_collides_with_box(self, box):
        my_box = self.get_bounding_box()
        xy_rect1 = (my_box[0], my_box[1], my_box[3], my_box[4])
        xy_rect2 = (box[0], box[1], box[3], box[4])
        if not util.rect_contains(xy_rect1, xy_rect2):
            return False
        xz_rect1 = (my_box[0], my_box[2], my_box[3], my_box[5])
        xz_rect2 = (box[0], box[2], box[3], box[5])
        if not util.rect_contains(xz_rect1, xz_rect2):
            return False
        return True

    def move(self, dxyz) -> 'Entity':
        self.xyz = util.add(self.xyz, dxyz)
        return self

    def is_liftable(self):
        return False

    def _normalize(self) -> 'Entity':
        box = self._calc_bounding_box(absolute=False)
        dx, dy, dz, *_ = box
        if dx != 0 or dy != 0 or dz != 0:
            cells = set()
            for (cx, cy, cz) in self.cells:
                cells.add((cx - dx, cy - dy, cz - dz))
            self.cells = cells
            self.xyz = util.add(self.xyz, (dx, dy, dz))
        self._box = (0, 0, 0, box[3], box[4], box[5])
        return self

    def rotate(self, cw_cnt=1, rel_pivot_pt=None) -> 'Entity':
        """Rotates the entity in the xy plane.

        If none is proved, the center of the entity's bounding box is used as the pivot.
        a---*---*---*   a---*---*  = (0, 0)
        | b |   |   |   | b |   |  = (0.5, 0.5)
        *---c---*---*   *---c---*  = (1, 1)
        |   | d |   |   |   | d |  = (1.5, 1.5)
        *---*---e---*   *---*---e  = (2, 2)
        |   |   |   |
        *---*---*---*
        """
        if rel_pivot_pt is None:
            box = self.get_bounding_box()
            if box[3] % 2 == box[4] % 2:
                rel_pivot_pt = (box[3] / 2, box[4] / 2)
            else:
                rel_pivot_pt = ((box[3] - 1) // 2 + 0.5, (box[4] - 1) // 2 + 0.5)
        else:
            good_pivot = False
            if int(rel_pivot_pt[0]) == rel_pivot_pt[0] and int(rel_pivot_pt[1]) == rel_pivot_pt[1]:
                good_pivot = True
            elif (int(rel_pivot_pt[0] + 0.5) == rel_pivot_pt[0] + 0.5 and
                  int(rel_pivot_pt[1] + 0.5) == rel_pivot_pt[1] + 0.5):
                good_pivot = True

            if not good_pivot:
                raise ValueError(f"bad pivot: {rel_pivot_pt}")

        cells = self.cells
        for _ in range(0, cw_cnt % 4):
            new_cells = set()
            for c in cells:
                x, y, z = c[0] + 0.5, c[1] + 0.5, c[2]
                new_x = rel_pivot_pt[1] - y + rel_pivot_pt[0]
                new_y = x - rel_pivot_pt[0] + rel_pivot_pt[1]
                new_cells.add((new_x - 0.5, new_y - 0.5, z))
            cells = new_cells

        self.cells = cells
        self._normalize()

        return self

    def __repr__(self):
        return f"{type(self).__name__}({self.xyz}, uid={self.uid}, " \
               f"cells={len(self.cells)}, " \
               f"box={self.get_bounding_box(absolute=False)}, " \
               f"com={self.get_center_of_mass(absolute=False)})"

    def __str__(self):
        res = [repr(self)]
        box = self.get_bounding_box()

        titles = []
        data = []
        for z_idx in range(0, box[5]):
            z = box[2] + z_idx
            titles.append(f"z={z}")
            for y_idx in range(0, box[4]):
                y = box[1] + y_idx
                line = []
                for x_idx in range(0, box[3]):
                    x = box[0] + x_idx
                    if (x, y, z) in self:
                        line.append("X")
                    else:
                        line.append(".")
                data.append("".join(line))

        spacer = 2
        item_width = max([8, box[3], 6 + len(str(box[2] + box[5]))])
        item_width += item_width % 2

        z_title_items = []
        for title in titles:
            z_title_items.append(f"{title}{(' ' * (item_width - (len(title))))}")
        res.append((' ' * spacer).join(z_title_items))

        for row in range(box[4]):
            data_items = []
            for z_idx in range(0, box[5]):
                data_str = data[z_idx * box[4] + row]
                if len(data_str) < item_width:
                    data_str += '.' * (item_width - len(data_str))
                data_items.append(data_str)
            res.append((' ' * spacer).join(data_items))

        return "\n".join(res)


class Block(Entity):

    def __init__(self, xyz, cells, color=(255, 255, 255), liftable=False, uid=None):
        super().__init__(xyz, cells, uid=uid)
        self.color = color
        self.liftable = liftable

    def is_liftable(self):
        return self.liftable

    def get_debug_color(self):
        return self.color


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Forklift(Entity):

    def __init__(self, xyz, direction=DIRECTIONS[0], uid=None):
        super().__init__(xyz, [(0, 0, z) for z in range(0, 6)], uid=uid)
        self.direction = direction
        self.fork_z = 0

    def get_direction(self):
        return self.direction

    def get_fork_xyz(self, absolute=True):
        x, y, z = self.xyz
        dirx, diry = self.get_direction()
        if absolute:
            return (x + dirx, y + diry, z + self.fork_z)
        else:
            return (dirx, diry, self.fork_z)

    def rotate(self, cw_cnt=1, rel_pivot_pt=None) -> 'Forklift':
        super().rotate(cw_cnt=1, rel_pivot_pt=None)
        dir_idx = DIRECTIONS.index(self.get_direction())
        self.direction = DIRECTIONS[(dir_idx + cw_cnt) % 4]
        return self


class World:

    def __init__(self):
        self.entities = set()
        self.terrain = {}

    def add_entity(self, ent):
        if ent in self.entities:
            self.entities.remove(ent)  # in case we're updating a stale ent
        self.entities.add(ent)

    def all_entities(self, cond=None):
        for ent in self.entities:
            if cond is None or cond(ent):
                yield ent

    def all_entities_at(self, xyz, cond=None) -> typing.Generator[Entity, None, None]:
        for ent in self.entities:
            if xyz in ent and (cond is None or cond(ent)):
                yield ent

    def all_entities_in_box(self, box, cond=None) -> typing.Generator[Entity, None, None]:
        for ent in self.entities:
            if ent.collides_with_box(box) and (cond is None or cond(ent)):
                yield ent


def build_sample_world():
    w = World()
    flift = (1, 1), (1, 0)
    holes = {(5, 0), (5, 1), (6, 1), (7, 1), (6, 2), (7, 2), (6, 3), (7, 3)}
    boxes = [(1, 0), (2, 0), (4, 0), (0, 3)]
    planks = [[(0, 0 + i) for i in range(3)],
              [(0 + i, 4) for i in range(3)],
              [(3 + i, 3) for i in range(5)]]
    for x in range(0, 8):
        for y in range(0, 5):
            if (x, y) not in holes:
                w.terrain[(x, y)] = 0

    w.add_entity(Forklift((*flift[0], 0), flift[1]))
    for b in boxes:
        w.add_entity(Block((0, 0, 0), [(*b, z) for z in range(0, 8)], color=(110, 112, 92), liftable=True))
    for plist in planks:
        w.add_entity(Block((0, 0, 0), [(p[0], p[1], 0) for p in plist], color=(164, 153, 131), liftable=True))

    return w


if __name__ == "__main__":
    ent = Entity((0, 0, 0), [(0, 0, 0), (1, 0, 0)])
    print(ent)
    print(ent.rotate(1))
    print(ent.collides(ent.copy().rotate(1)))
