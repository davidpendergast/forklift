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

    def get_cells(self, absolute=True):
        x, y, z = self.xyz if absolute else (0, 0, 0)
        return [(c[0] + x, c[1] + y, c[2] + z) for c in self.cells]

    def get_weight(self):
        return len(self.cells)

    def get_bounding_cube(self, absolute=True):
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

    def get_debug_color(self):
        raise (55, 55, 55)

    def copy(self, keep_uid=True) -> 'Entity':
        raise NotImplementedError

    def __contains__(self, xyz):
        return util.sub(xyz, self.xyz) in self.cells

    def move(self, dxyz) -> 'Entity':
        self.xyz = util.add(self.xyz, dxyz)
        return self

    def _normalize(self) -> 'Entity':
        box = self.get_bounding_cube()
        dx, dy, dz, *_ = box
        if dx != 0 or dy != 0 or dz != 0:
            cells = set()
            for (cx, cy, cz) in self.cells:
                cells.add((cx - dx, cy - dy, cz - dz))
            self.cells = cells
            self.xyz = util.add(self.xyz, (dx, dy, dz))
        return self

    def rotate(self, cw_cnt=1, rel_pivot_pt=None) -> 'Entity':
        """Rotates the entity in place, updating both the cells and the offset if necessary.

        The pivot point refers to voxel centers, and only needs x and y components.
        If none is proved, the center of the entity's bounding box is used as the pivot.
        *---*---*---*   *---*---*
        | a |   |   |   | a |   |  = (0, 0)
        *---b---*---*   *---b---*  = (0.5, 0.5)
        |   | c |   |   |   | c |  = (1, 1)
        *---*---d---*   *---*---d  = (1.5, 1.5)
        |   |   |   |
        *---*---*---*
        """
        box = self.get_bounding_cube()

        if rel_pivot_pt is None:
            if box[3] % 2 == box[4] % 2:
                rel_pivot_pt = ((box[3] - 1) / 2, (box[4] - 1) / 2)
            else:
                rel_pivot_pt = ((box[3] - 1) // 2, (box[4] - 1) // 2)
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
                x, y, z = c
                new_x = rel_pivot_pt[1] - y + rel_pivot_pt[0]
                new_y = x - rel_pivot_pt[0] + rel_pivot_pt[1]
                new_cells.add((new_x, new_y, z))
            cells = new_cells

        self.cells = cells
        self._normalize()

        return self

    def __repr__(self):
        return f"{type(self).__name__}({self.xyz}, uid={self.uid}, " \
               f"cells={len(self.cells)}, " \
               f"box={self.get_bounding_cube(absolute=False)})"

    def __str__(self):
        res = [repr(self)]
        box = self.get_bounding_cube()

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


class World:

    def __init__(self):
        self.entities = []
        self.terrain = {}

    def get_at(self, xyz, cond=None) -> typing.List[Entity]:
        pass


if __name__ == "__main__":
    ent = Entity((0, 0, 0), [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 3, 0)])
    print(ent)
    print(ent.rotate(1))