import math
import typing

import src.utils.util as util
import src.engine.globaltimer as globaltimer
import src.utils.colorutils as colorutils

import src.engine.sprites as sprites
import src.engine.threedee as threedee
import src.engine.spritesheets as spritesheets
import src.engine.renderengine as renderengine

import src.game.spriteref as spriteref

_ENT_ID = 0


def _next_ent_id():
    global _ENT_ID
    _ENT_ID += 1
    return _ENT_ID - 1


class Entity:

    DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, xyz, cells, direction=DIRECTIONS[0], uid=None):
        self.uid = uid if uid is not None else _next_ent_id()
        self.xyz = xyz
        self.direction = direction
        self.cells = set(cells)
        self._box = None
        self._normalize()

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.uid == other.uid
        else:
            return self.uid == other  # so we can be equal to ints

    def __hash__(self):
        return hash(self.uid)

    def get_xyz(self):
        return self.xyz

    def set_xyz(self, xyz) -> 'Entity':
        self.xyz = xyz
        return self

    def get_direction(self) -> typing.Tuple[int, int]:
        return self.direction

    def get_direction_idx(self) -> int:
        return Entity.DIRECTIONS.index(self.direction)

    def get_cells(self, absolute=True) -> typing.Set[typing.Tuple[int, int, int]]:
        x, y, z = self.xyz if absolute else (0, 0, 0)
        return set((c[0] + x, c[1] + y, c[2] + z) for c in self.cells)

    def get_top_cells(self, add_z=0, absolute=True, include_holes=True):
        max_z_for_xy = {}
        top_edge_cells = set()
        for (x, y, z) in self.get_cells(absolute=absolute):
            if (x, y) not in max_z_for_xy:
                max_z_for_xy[(x, y)] = z
            else:
                max_z_for_xy[(x, y)] = max(max_z_for_xy[(x, y)], z)
            if not self.contains_cell((x, y, z+1), absolute=absolute):
                top_edge_cells.add((x, y, z))

        if include_holes:
            return set((x, y, z + add_z) for (x, y, z) in top_edge_cells)
        else:
            return set((x, y, max_z_for_xy[(x, y)] + add_z) for (x, y) in max_z_for_xy)

    def get_bottom_cells(self, add_z=0, absolute=True, include_holes=True):
        min_z_for_xy = {}
        bottom_edge_cells = set()
        for (x, y, z) in self.get_cells(absolute=absolute):
            if (x, y) not in min_z_for_xy:
                min_z_for_xy[(x, y)] = z
            else:
                min_z_for_xy[(x, y)] = min(min_z_for_xy[(x, y)], z)
            if not self.contains_cell((x, y, z - 1), absolute=absolute):
                bottom_edge_cells.add((x, y, z))

        if include_holes:
            return set((x, y, z + add_z) for (x, y, z) in bottom_edge_cells)
        else:
            return set((x, y, min_z_for_xy[(x, y)] + add_z) for (x, y) in min_z_for_xy)

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

    def get_bounding_box(self, absolute=True, axes='xyz'):
        if self._box is None:
            self._normalize()

        if absolute:
            res = (self._box[0] + self.xyz[0],
                   self._box[1] + self.xyz[1],
                   self._box[2] + self.xyz[2],
                   self._box[3], self._box[4], self._box[5])
        else:
            res = self._box

        if axes == 'xyz':
            return res
        else:
            return tuple(res[(ord(i) - ord('x')) % 3] for i in axes) \
                   + tuple(res[3 + ((ord(i) - ord('x')) % 3)] for i in axes)

    def get_max(self, absolute=True, axes='xyz'):
        bb = self.get_bounding_box(absolute=absolute, axes=axes)
        return tuple(bb[i] + bb[len(bb) // 2 + i] for i in range(len(bb) // 2))

    def get_min(self, absolute=True, axes='xyz'):
        bb = self.get_bounding_box(absolute=absolute, axes=axes)
        return tuple(bb[i] for i in range(len(bb) // 2))

    def get_base_color(self):
        return (55, 55, 55)

    def get_color(self):
        return self.get_base_color()

    def copy(self, keep_uid=True) -> 'Entity':
        if type(self) is Entity:
            return Entity(self.xyz, self.cells, direction=self.direction, uid=self.uid if keep_uid else None)
        else:
            raise NotImplementedError()

    def __contains__(self, xyz):
        xyz_int = (int(xyz[0]) - self.xyz[0],
                   int(xyz[1]) - self.xyz[1],
                   int(xyz[2]) - self.xyz[2])
        return xyz_int in self.cells

    def contains_cell(self, xyz, absolute=True):
        if absolute:
            return xyz in self
        else:
            return xyz in self.cells

    def contains_any(self, xyz_list, absolute=True):
        for xyz in xyz_list:
            if self.contains_cell(xyz, absolute=absolute):
                return True
        return False

    def contains_all(self, xyz_list, absolute=True):
        for xyz in xyz_list:
            if not self.contains_cell(xyz, absolute=absolute):
                return False
        return True

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

    def rotate(self, cw_cnt=1, rel_pivot_pt=None, abs_pivot_pt=None) -> 'Entity':
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
        if abs_pivot_pt is not None:
            if rel_pivot_pt is not None:
                raise ValueError("Cannot have both a relative and absolute pivot point")
            rel_pivot_pt = util.sub(abs_pivot_pt[:2], self.xyz[:2])

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
                new_cells.add((int(new_x - 0.5), int(new_y - 0.5), z))
            cells = new_cells

        self.cells = cells
        self._normalize()

        dir_idx = Entity.DIRECTIONS.index(self.get_direction())
        self.direction = Entity.DIRECTIONS[(dir_idx + cw_cnt) % 4]

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

    def __init__(self, xyz, cells, color=(255, 255, 255), direction=Entity.DIRECTIONS[0], liftable=True, uid=None):
        super().__init__(xyz, cells, direction=direction, uid=uid)
        self.color = color
        self.liftable = liftable

    def is_liftable(self):
        return self.liftable

    def get_base_color(self):
        return self.color

    def copy(self, keep_uid=True) -> 'Block':
        return Block(self.get_xyz(), self.get_cells(absolute=False),
                     direction=self.get_direction(),
                     color=self.color,
                     liftable=self.liftable,
                     uid=self.uid if keep_uid else None)

    def is_win(self) -> bool:
        return False


class WinBlock(Block):

    def __init__(self, xyz, cells, color=(255, 255, 0), direction=Entity.DIRECTIONS[0], liftable=True, uid=None):
        super().__init__(xyz, cells, color=color, direction=direction, liftable=liftable, uid=uid)
        self._max_darken = 0.25
        self._darken_period = 1  # second

    def get_color(self):
        color = super().get_color()
        t = globaltimer.get_elapsed_time() / 1000
        darken_pcnt = self._max_darken * 0.5 * (1 + math.sin(t * 2 * math.pi / self._darken_period))
        return colorutils.to_int(colorutils.darker(colorutils.to_float(color), darken_pcnt))

    def is_win(self) -> bool:
        return True

    def copy(self, keep_uid=True) -> 'WinBlock':
        return WinBlock(self.get_xyz(), self.get_cells(absolute=False),
                        direction=self.get_direction(),
                        color=self.color,
                        liftable=self.liftable,
                        uid=self.uid if keep_uid else None)


class Forklift(Entity):

    def __init__(self, xyz, direction=Entity.DIRECTIONS[0], fork_z=0, fork_z_bounds=(0, 8), uid=None):
        super().__init__(xyz, [(0, 0, z) for z in range(0, 6)], direction=direction, uid=uid)
        self.fork_z_bounds = fork_z_bounds
        self.fork_z = util.bound(fork_z, *fork_z_bounds)

    def get_base_color(self):
        return (224, 224, 64)

    def get_fork_xyz(self, absolute=True):
        x, y, z = self.xyz
        dirx, diry = self.get_direction()
        if absolute:
            return (x + dirx, y + diry, z + self.fork_z)
        else:
            return (dirx, diry, self.fork_z)

    def move_fork(self, df, safe=True):
        self.fork_z += df
        if safe:
            self.fork_z = util.bound(self.fork_z, *self.fork_z_bounds)
        if not self.fork_z_bounds[0] <= self.fork_z <= self.fork_z_bounds[1]:
            raise ValueError(f"fork height is out of range ({self.fork_z} not in {self.fork_z_bounds})")

        return self

    def rotate(self, cw_cnt=1, rel_pivot_pt=None, abs_pivot_pt=None) -> 'Forklift':
        super().rotate(cw_cnt=cw_cnt, rel_pivot_pt=rel_pivot_pt, abs_pivot_pt=abs_pivot_pt)
        return self

    def copy(self, keep_uid=True) -> 'Forklift':
        return Forklift(self.get_xyz(),
                        direction=self.get_direction(),
                        fork_z=self.fork_z,
                        fork_z_bounds=self.fork_z_bounds,
                        uid=self.uid if keep_uid else None)


class LiftableEntityStack:

    def __init__(self):
        self.entities = set()
        self.above = {}  # keys are above values
        self.below = {}  # keys are below values

    def __contains__(self, item):
        return item in self.entities

    def __iter__(self) -> typing.Iterator[Entity]:
        return (e for e in self.entities)

    def __repr__(self):
        return f"{type(self).__name__}(\n\tentities={self.entities}, \n\tabove={self.above}, \n\tbelow={self.below})"

    def is_balanced(self, support_points):
        pass

    def add(self, ent) -> bool:
        if ent in self.entities:
            return False
        else:
            self.entities.add(ent)
            self.above[ent] = set()
            self.below[ent] = set()
            return True

    def set_above(self, ent_above, ent_below) -> bool:
        did_change = self.add(ent_above)
        did_change |= self.add(ent_below)

        if ent_below not in self.above[ent_above]:
            self.above[ent_above].add(ent_below)
            self.below[ent_below].add(ent_above)
            did_change = True

        return did_change

    def _traverse(self, start, find_neighbors_func, exclude=(), ignore=()) -> typing.Generator[Entity, None, None]:
        seen = set()
        q = []
        if isinstance(start, Entity):
            q.append(start)
            seen.add(start)
        else:
            q.extend(start)
            seen.update(start)
        while len(q) > 0:
            cur = q.pop(-1)
            for e in find_neighbors_func(cur):
                if e in seen or e in exclude:
                    continue
                else:
                    seen.add(e)
                    q.append(e)
                    if e not in ignore:
                        yield e

    def all_above_recurse(self, start, exclude=(), ignore=()) -> typing.Generator[Entity, None, None]:
        """Recursively finds all stack entities that are above the given entity or collection of entities.
            start: an Entity or collection of Entities.
            exclude: collection of Entities to pretend don't exist.
            ignore: collection of Entities to recurse upon but not yield.
        """
        return self._traverse(start, lambda x: self.below[x], exclude=exclude, ignore=ignore)

    def all_below_recurse(self, start, exclude=(), ignore=()) -> typing.Generator[Entity, None, None]:
        """Recursively finds all stack entities that are below the given entity or collection of entities.
            start: an Entity or collection of Entities.
            exclude: collection of Entities to pretend don't exist.
            ignore: collection of Entities to recurse upon but not yield.
        """
        return self._traverse(start, lambda x: self.above[x], exclude=exclude, ignore=ignore)


class ForkliftActionHandler:

    @staticmethod
    def get_stack_above(xyz, world_state: 'World', cond=None):
        res = LiftableEntityStack()

        q = set()
        processed = set()

        # find 'base' entities
        for ent in world_state.all_entities_at(xyz, cond=cond):
            q.add(ent)
            res.add(ent)

        while len(q) > 0:
            ent = q.pop()
            processed.add(ent)
            for cell in ent.get_top_cells(add_z=1):
                for ent_above in world_state.all_entities_at(cell, cond=cond):
                    res.set_above(ent_above, ent)
                    if ent_above not in processed:
                        q.add(ent_above)

        return res

    @staticmethod
    def move_fork(forklift: Forklift, steps, world_state: 'World', log=True, all_or_none=True) -> typing.Optional['AbstractWorldMutation']:
        if steps == 0:
            return WorldMutation()

        muts = []
        for _ in range(abs(steps)):
            if steps > 0:
                mut = ForkliftActionHandler.raise_fork(forklift, world_state, log=log)
            else:
                mut = ForkliftActionHandler.lower_fork(forklift, world_state, log=log)

            if mut is not None:
                muts.append(mut)
            elif all_or_none:
                return None
            else:
                break

        if len(muts) == 0:
            return None
        else:
            return CompositeWorldMutation(muts=muts)

    @staticmethod
    def raise_fork(forklift: Forklift, world_state: 'World', log=True) -> typing.Optional['WorldMutation']:
        if forklift.fork_z >= forklift.fork_z_bounds[1]:
            return None
        else:
            stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state,
                                                                  cond=lambda e: e.is_liftable())
            # check to see if the stack is blocked from above
            mut = WorldMutation()
            mut.reg(forklift, forklift.copy().move_fork(1, safe=False))
            for stack_ent in stack_on_fork.entities:
                if stack_ent == forklift:
                    if log: print("INFO: you can't lift yourself ._.")
                    return None

                for col_ent in world_state.all_entities_colliding_with(stack_ent, offs=(0, 0, 1),
                                                                       cond=lambda e: e not in stack_on_fork):
                    if log: print(f"INFO: stack will collide if lifted ({stack_ent} -> {col_ent})")
                    return None

                mut.reg(stack_ent, stack_ent.copy().move((0, 0, 1)))

            return mut

    @staticmethod
    def lower_fork(forklift: Forklift, world_state: 'World', log=True) -> typing.Optional['WorldMutation']:
        if forklift.fork_z <= forklift.fork_z_bounds[0]:
            return None
        else:
            stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state,
                                                                  cond=lambda e: e.is_liftable())
            # ensure fork can lower
            for ent in world_state.all_entities_below_fork(forklift.get_fork_xyz()):
                if ent not in stack_on_fork:
                    if log: print(f"INFO: fork is blocked by {ent}")
                    return None

            # find ents in stack that can't be lowered (because they're directly blocked)
            ents_blocked = set()
            for stack_ent in stack_on_fork:
                for cell in stack_ent.get_bottom_cells(add_z=-1):
                    for _ in world_state.all_entities_at(cell, cond=lambda e: e not in stack_on_fork):
                        ents_blocked.add(stack_ent)
                        break

            # find ents that are blocked indirectly
            q = set()
            q.update(ents_blocked)
            while len(q) > 0:
                stack_ent = q.pop()
                for ent_above in stack_on_fork.below[stack_ent]:
                    if ent_above not in ents_blocked:
                        q.add(ent_above)
                        ents_blocked.add(ent_above)

            mut = WorldMutation()
            mut.reg(forklift, forklift.copy().move_fork(-1, safe=False))

            # finally lower the unblocked ents
            for stack_ent in stack_on_fork:
                if stack_ent not in ents_blocked:
                    mut.reg(stack_ent, stack_ent.copy().move((0, 0, -1)))

            return mut

    @staticmethod
    def rotate_forklift(forklift: Forklift, cw: bool, world_state: 'World', all_or_none=True, log=True) \
            -> typing.Optional['WorldMutation']:

        stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state)

        cw_cnt = 1 if cw else -1
        abs_pivot_pt = util.add(forklift.get_xyz()[:2], (0.5, 0.5))
        new_forklift = forklift.copy().rotate(cw_cnt, abs_pivot_pt=abs_pivot_pt)

        if not ForkliftActionHandler.is_valid_position_for_forklift(new_forklift, world_state,
                                                                    stack_on_fork=stack_on_fork):
            return None

        directly_supported = set()
        for stack_ent in stack_on_fork:
            if world_state.is_supported_from_below(stack_ent, cond=lambda e: e not in stack_on_fork):
                directly_supported.add(stack_ent)

        all_supported = set(stack_on_fork.all_above_recurse(directly_supported)) | directly_supported

        if len(all_supported) == 0:
            stack_to_rotate = stack_on_fork
        elif all_supported == stack_on_fork.entities:
            stack_to_rotate = None
        else:
            stack_to_rotate = LiftableEntityStack()
            for stack_ent in stack_on_fork:
                if stack_ent not in all_supported:
                    stack_to_rotate.add(stack_ent)
                    for below_ent in stack_on_fork.above[stack_ent]:
                        if below_ent not in all_supported:
                            stack_to_rotate.set_above(stack_ent, below_ent)
                    for above_ent in stack_on_fork.below[stack_ent]:
                        if above_ent not in all_supported:
                            stack_to_rotate.set_above(above_ent, stack_ent)

        mut = WorldMutation()
        mut.reg(forklift, new_forklift)

        if stack_to_rotate is not None:
            # rotate the stack
            for stack_ent in stack_to_rotate:
                rotated_ent = stack_ent.copy().rotate(cw_cnt, abs_pivot_pt=abs_pivot_pt)
                for world_ent in world_state.all_entities_colliding_with(rotated_ent,
                                                                         cond=lambda e: e not in stack_to_rotate):
                    if log: print(f"INFO: stack ent will collide if rotated ({stack_ent} -> {world_ent})")
                    return None
                mut.reg(stack_ent, rotated_ent)

        return mut

    @staticmethod
    def is_valid_position_for_forklift(forklift: Forklift, world_state: 'World', stack_on_fork=None, log=True) -> bool:
        if stack_on_fork is None:
            stack_on_fork = ()

        cond = lambda e: e not in stack_on_fork and e != forklift

        for ent in world_state.all_entities_colliding_with(forklift, cond=cond):
            if log: print(f"INFO: forklift is blocked by {ent}")
            return False
        for ent in world_state.all_entities_intersected_by_fork(forklift.get_fork_xyz(), cond=cond):
            if log: print(f"INFO: fork would skewer {ent}")
            return False
        if world_state.get_terrain_height(forklift.get_xyz()[:2]) != forklift.get_xyz()[2]:
            supported = False
            for _ in world_state.all_entities_at(util.add(forklift.get_xyz(), (0, 0, -1)), cond=cond):
                supported = True
                break
            if not supported:
                if log: print(f"INFO: no ground for forklift at ({forklift.get_xyz()})")
                return False
        return True

    @staticmethod
    def move_forklift(forklift: Forklift, forward: bool, world_state: 'World', log=True) \
            -> typing.Optional['WorldMutation']:

        # TODO should the forklift be able to 'push' stuff?
        #  - no, OSHA
        stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state)
        move_xy = forklift.get_direction() if forward else util.negate(forklift.get_direction())

        move_z = None
        for dz in (0, -1, 1):
            new_forklift = forklift.copy().move((*move_xy, dz))
            if ForkliftActionHandler.is_valid_position_for_forklift(new_forklift, world_state,
                                                                    stack_on_fork=stack_on_fork, log=False):
                move_z = dz
                break

        if move_z is None:  # can't move there
            if log:
                # just to make it log
                ForkliftActionHandler.is_valid_position_for_forklift(forklift.copy().move((*move_xy, 0)),
                                                                     world_state, stack_on_fork=stack_on_fork, log=log)
            return None

        move_xyz = (*move_xy, move_z)

        mut = WorldMutation()
        mut.reg(forklift, new_forklift)

        blocked_stack_ents = set()

        # ensure stack can be moved
        for stack_ent in stack_on_fork:
            for _ in world_state.all_entities_colliding_with(stack_ent, offs=move_xyz,
                                                             cond=lambda e: e not in stack_on_fork and e != forklift):
                blocked_stack_ents.add(stack_ent)
                break

        # entities on top of blocked ents shouldn't move either
        # TODO this logic should take balance points into account
        keep_checking = True
        while keep_checking:
            keep_checking = False
            for stack_ent in stack_on_fork:
                if stack_ent in blocked_stack_ents:
                    continue
                else:
                    for ent_below in stack_on_fork.above[stack_ent]:
                        if ent_below in blocked_stack_ents:
                            blocked_stack_ents.add(stack_ent)
                            keep_checking = True
                            break

        moved_ents = set()
        moved_ents.add(mut.updates[forklift][1])

        # move things that aren't blocked
        for moved_ent in stack_on_fork:
            if moved_ent not in blocked_stack_ents:
                mut.reg(moved_ent, moved_ent.copy().move(move_xyz))
                moved_ents.add(mut.updates[moved_ent][1])

        # ensure things that don't move are supported and won't collide with anything that did move
        for blocked_ent in blocked_stack_ents:
            supported = False

            # see if it's supported by or colliding with anything that moved
            for moved_ent in moved_ents:
                if blocked_ent.collides(moved_ent):
                    if log: print(f"INFO: blocked ent {blocked_ent} will collide with moved ent {moved_ent}")
                    return None
                if (not supported and not isinstance(moved_ent, Forklift)
                        and moved_ent.contains_any(blocked_ent.get_bottom_cells(add_z=-1))):
                    supported = True

            # see if it's supported by anything in the world (including other things that didn't move).
            if not supported:
                for cell_below in blocked_ent.get_bottom_cells(add_z=-1):
                    if world_state.get_terrain_height(cell_below[:2]) == cell_below[2] + 1:
                        supported = True
                        break
                    for _ in world_state.all_entities_at(cell_below, cond=lambda e: e not in moved_ents):
                        supported = True
                        break
                    if supported:
                        break

            if not supported:
                if log: print(f"INFO: blocked ent won't be supported {blocked_ent}.")
                return None

        return mut


class AbstractWorldMutation:

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def apply(self, world_state: 'World'):
        raise NotImplementedError()

    def undo(self, world_state: 'World'):
        raise NotImplementedError()

    def get_entity_change(self, ent: typing.Union[int, Entity]) \
            -> typing.Tuple[typing.Optional[Entity], typing.Optional[Entity]]:
        raise NotImplementedError()


class WorldMutation(AbstractWorldMutation):

    def __init__(self):
        self.updates = {}  # ent_id -> (old copy of Entity, new copy of Entity)

    def is_empty(self):
        return len(self.updates) == 0

    def reg(self, old_ent: Entity, new_ent: Entity):
        if old_ent is None and new_ent is None:
            return
        elif old_ent is None:
            self.updates[new_ent.uid] = (None, new_ent)  # addition
        elif new_ent is None:
            self.updates[old_ent.uid] = (old_ent, None)  # deletion
        elif old_ent.uid != new_ent.uid:
            self.updates[old_ent.uid] = (old_ent, None)  # transitioned into another
            self.updates[new_ent.uid] = (None, new_ent)
        else:
            self.updates[new_ent.uid] = (old_ent, new_ent)  # changed

    def apply(self, world_state: 'World'):
        for ent_id in self.updates:
            old_ent, new_ent = self.updates[ent_id]
            if old_ent is not None:
                world_state.remove_entity(old_ent)
            if new_ent is not None:
                world_state.add_entity(new_ent)

    def undo(self, world_state: 'World'):
        for (old_ent, new_ent) in self.updates.values():
            if new_ent is not None:
                world_state.remove_entity(new_ent)
            if old_ent is not None:
                world_state.add_entity(old_ent)

    def get_entity_change(self, ent: typing.Union[int, Entity]) \
            -> typing.Tuple[typing.Optional[Entity], typing.Optional[Entity]]:
        ent_id = ent if isinstance(ent, int) else ent.uid
        if ent_id in self.updates:
            return self.updates[ent_id]
        else:
            return None, None


class CompositeWorldMutation(AbstractWorldMutation):

    def __init__(self, muts=()):
        # TODO should just merge these immediately
        self.muts: typing.List[AbstractWorldMutation] = list(muts)

    def is_empty(self):
        for mut in self.muts:
            if not mut.is_empty():
                return False
        return True

    def add(self, mut):
        if mut is not None:
            self.muts.append(mut)

    def apply(self, world_state: 'World'):
        for m in self.muts:
            m.apply(world_state)

    def undo(self, world_state: 'World'):
        for m in reversed(self.muts):
            m.undo(world_state)

    def get_entity_change(self, ent: typing.Union[int, Entity]) \
            -> typing.Tuple[typing.Optional[Entity], typing.Optional[Entity]]:
        old_ent, new_ent = None, None
        for mut in self.muts:
            old_new = mut.get_entity_change(ent)
            if old_ent is None and old_new[0] is not None:
                old_ent = old_new[0]
            if old_new[1] is not None:
                new_ent = old_new[1]
        return old_ent, new_ent


class World:

    def __init__(self):
        self.entities = set()
        self.terrain = {}

    def add_entity(self, ent):
        if ent is None or not isinstance(ent, Entity):
            raise TypeError(f"Expected an Entity, instead got: {ent} ({type(ent).__name__})")
        if ent in self.entities:
            self.entities.remove(ent)  # in case we're updating a stale ent
        self.entities.add(ent)

    def remove_entity(self, ent):
        if ent is None or not isinstance(ent, Entity):
            raise TypeError(f"Expected an Entity, instead got: {ent} ({type(ent).__name__})")
        if ent not in self.entities:
            print(f"WARN: Tried to remove a non-existent Entity: {ent}")
        else:
            self.entities.remove(ent)

    def all_entities(self, cond=None) -> typing.Generator[Entity, None, None]:
        for ent in self.entities:
            if cond is None or cond(ent):
                yield ent

    def get_forklift(self) -> typing.Optional[Forklift]:
        for ent in self.all_entities(cond=lambda e: isinstance(e, Forklift)):
            return ent
        return None

    def all_entities_at(self, xyz, cond=None) -> typing.Generator[Entity, None, None]:
        for ent in self.entities:
            if xyz in ent and (cond is None or cond(ent)):
                yield ent

    def all_entities_in_box(self, box, cond=None) -> typing.Generator[Entity, None, None]:
        for ent in self.entities:
            if ent.collides_with_box(box) and (cond is None or cond(ent)):
                yield ent

    def all_entities_colliding_with(self, ent: Entity, offs=(0, 0, 0), xyz_override=None, cond=None) \
            -> typing.Generator['Entity', None, None]:
        seen = set()
        seen.add(ent)  # prevent self-collisions
        xyz = util.add(ent.get_xyz() if xyz_override is None else xyz_override, offs)
        for rel_c in ent.get_cells(absolute=False):
            abs_c = util.add(xyz, rel_c)
            for other in self.all_entities_at(abs_c, cond=cond):
                if other not in seen:
                    yield other
                seen.add(other)

    def all_entities_on_fork(self, fork_xyz, cond=None) -> typing.Generator['Entity', None, None]:
        return self.all_entities_at(fork_xyz, cond=cond)

    def all_entities_below_fork(self, fork_xyz, cond=None) -> typing.Generator['Entity', None, None]:
        below_fork_xyz = util.add(fork_xyz, (0, 0, -1))
        return self.all_entities_at(below_fork_xyz, cond=cond)

    def all_entities_intersected_by_fork(self, fork_xyz, cond=None) -> typing.Generator['Entity', None, None]:
        """Entities that are both above and below the specified fork position."""
        ents_above = set()
        for ent in self.all_entities_on_fork(fork_xyz, cond=cond):
            ents_above.add(ent)
        for ent in self.all_entities_below_fork(fork_xyz, cond=cond):
            if ent in ents_above:
                yield ent

    def get_terrain_height(self, xy, or_else=-1000) -> int:
        if xy in self.terrain:
            return self.terrain[xy]
        else:
            return or_else

    def is_supported_from_below(self, ent: Entity, cond=None):
        for cell in ent.get_bottom_cells():
            if self.get_terrain_height(cell[:2], or_else=None) == cell[2]:
                return True  # supported by terrain
        for _ in self.all_entities_colliding_with(ent, offs=(0, 0, -1), cond=cond):
            return True  # supported by another entity
        return False


def build_sample_world(idx=0):
    levels = [build_complex_world, build_single_block_world]
    return levels[idx % len(levels)]()


def build_complex_world():
    w = World()
    flift = (3, 1), (1, 0)
    holes = {(5, 0), (5, 1), (6, 1), (7, 1), (6, 2), (7, 2), (6, 3), (7, 3)}
    boxes = [(1, 0)]
    wins = [(2, 0)]
    half_boxes = [(4, 0), (0, 3)]
    pallet = [(5, 3), (2, 3)]
    planks = [[(0, 0 + i) for i in range(3)],
              [(0 + i, 4) for i in range(3)],
              [(4 + i, 4) for i in range(3)],
              [(0 + i, 0, 8) for i in range(3)]]
    for x in range(0, 8):
        for y in range(0, 5):
            if (x, y) not in holes:
                w.terrain[(x, y)] = 0

    w.add_entity(Forklift((*flift[0], 0), flift[1]))
    for b in boxes:
        w.add_entity(Block((0, 0, 0), [(*b, z) for z in range(0, 8)], color=(112, 112, 92), liftable=True))
    for win in wins:
        w.add_entity(WinBlock((0, 0, 0), [(*win, z) for z in range(0, 8)], liftable=True))
    for hb in half_boxes:
        w.add_entity(Block((0, 0, 0), [(*hb, z) for z in range(0, 4)], color=(128, 144, 160), liftable=True))
    for p in pallet:
        w.add_entity(Block((0, 0, 0), [(*p, z) for z in range(0, 2)], color=(196, 196, 196), liftable=True))
    for plist in planks:
        w.add_entity(Block((0, 0, 0), [(p[0], p[1], (p[2] if len(p) >= 3 else 0)) for p in plist],
                           color=(160, 160, 128), liftable=True))

    return w


def build_single_block_world():
    w = World()
    flift = (3, 1), (1, 0)
    holes = {(5, 0), (5, 1), (6, 1), (7, 1), (6, 2), (7, 2), (6, 3), (7, 3)}
    boxes = [(0, 0)]
    planks = []
    for x in range(0, 8):
        for y in range(0, 5):
            if (x, y) not in holes:
                w.terrain[(x, y)] = 0

    w.add_entity(Forklift((*flift[0], 0), flift[1]))
    for b in boxes:
        w.add_entity(Block((0, 0, 0), [(*b, z) for z in range(0, 8)], color=(110, 112, 92), liftable=True))
    for plist in planks:
        w.add_entity(Block((0, 0, 0), [(p[0], p[1], (p[2] if len(p) >= 3 else 0)) for p in plist],
                           color=(164, 153, 131), liftable=True))

    return w


class WorldRenderer:

    def __init__(self):
        self.cur_time = 0
        self.muts = []  # list of (start_time, end_time, reg_time, WorldMutation)

    def register_mutation(self, mut: AbstractWorldMutation, duration_ms: float, start_delay=0, clear_old=True):
        if clear_old:
            self.muts.clear()
        if mut is not None:
            start_time = self.cur_time + start_delay
            end_time = start_time + duration_ms
            if self.cur_time < end_time and start_time < end_time:
                self.muts.append((start_time, end_time, self.cur_time, mut))

    def update_muts(self):
        if len(self.muts) > 0:
            self.muts = [m for m in self.muts if m[1] > self.cur_time]

    def all_active_muts(self) -> typing.Generator[typing.Tuple[AbstractWorldMutation, float], None, None]:
        for m in self.muts:
            if m[0] <= self.cur_time <= m[1]:
                yield m[3], (self.cur_time - m[0]) / (m[1] - m[0])

    def get_active_mut(self) -> typing.Tuple[AbstractWorldMutation, float]:
        for mut, prog in reversed(list(self.all_active_muts())):
            return mut, prog  # TODO need to merge & combine overlapping muts
        return None, None

    def get_cur_entity(self, ent: Entity) -> typing.Tuple[Entity, Entity, float]:
        mut, prog = self.get_active_mut()
        if mut is None:
            return ent, ent, 0
        else:
            old_ent, new_ent = mut.get_entity_change(ent)
            if old_ent is None and new_ent is None:
                return ent, ent, 0
            else:
                return old_ent, new_ent, prog

    def update(self, world_state: World):
        self.cur_time += globaltimer.dt()
        self.update_muts()

    def all_sprites(self) -> typing.Generator[sprites.AbstractSprite, None, None]:
        pass


class WorldRenderer2D(WorldRenderer):

    CELL_SIZE = 64

    def __init__(self):
        super().__init__()
        self._terrain_sprites = {}  # (x, y) -> AbstractSprite
        self._entity_sprites = {}  # ent_id -> AbstractSprite

    def update(self, world_state: World):
        super().update(world_state)
        new_terrain_sprites = {}
        cs = WorldRenderer2D.CELL_SIZE
        xmin, xmax = float('inf'), -float('inf')
        ymin, ymax = float('inf'), -float('inf')

        for (x, y) in world_state.terrain.keys():
            xmin, xmax = util.update_bounds(xmin, xmax, x)
            ymin, ymax = util.update_bounds(ymin, ymax, y)
            rect = (x * cs, y * cs, cs, cs)
            if (x, y) in self._terrain_sprites:
                spr = self._terrain_sprites[(x, y)]
            else:
                spr = sprites.RectangleOutlineSprite(spriteref.LAYER_POLY)
            height = world_state.get_terrain_height((x, y))
            color = colorutils.lighter((0.33, 0.33, 0.33), 0.1 * height)
            spr = spr.update(new_rect=rect, new_color=color,
                             new_depth=-height, new_outline=cs // 16)
            new_terrain_sprites[(x, y)] = spr
        self._terrain_sprites = new_terrain_sprites

        new_entity_sprites = {}

        for ent in world_state.all_entities():
            color = colorutils.to_float(*ent.get_base_color())
            depth = -ent.get_max(axes='z')[0]
            rect = util.mult(ent.get_bounding_box(axes='xy'), cs)
            rect = util.rect_expand(rect, all_expand=-cs // 16)

            if ent.uid in self._entity_sprites:
                spr = self._entity_sprites[ent.uid]
            else:
                spr = sprites.ImageSprite.new_sprite(spriteref.LAYER_WORLD_2D)

            spr = spr.update(new_model=spritesheets.get_white_square_img(), new_x=rect[0], new_y=rect[1],
                             new_raw_size=rect[2:4], new_color=color, new_depth=depth)
            new_entity_sprites[ent.uid] = spr

            if isinstance(ent, Forklift):
                fork_id = f"{ent.uid}_fork"
                if fork_id in self._entity_sprites:
                    fork_spr = self._entity_sprites[fork_id]
                else:
                    fork_spr = sprites.ImageSprite.new_sprite(spriteref.LAYER_WORLD_2D)
                fork_rect = (ent.get_fork_xyz()[0], ent.get_fork_xyz()[1], 1, 1)
                fork_rect = util.rect_expand(util.mult(fork_rect, cs), all_expand=-cs // 16)
                fork_depth = -ent.get_fork_xyz()[2]
                fork_spr = fork_spr.update(new_model=spritesheets.get_white_square_img(),
                                           new_x=fork_rect[0], new_y=fork_rect[1],
                                           new_raw_size=fork_rect[2:4], new_depth=fork_depth,
                                           new_color=colorutils.darker(color))
                new_entity_sprites[fork_id] = fork_spr

        self._entity_sprites = new_entity_sprites

        if xmin != float('inf'):
            world_center = ((xmin + xmax + 1) * cs // 2, (ymin + ymax + 1) * cs // 2)
        else:
            world_center = (0, 0)

        for lay in renderengine.get_instance().get_layers(spriteref.world_2d_layer_ids()):
            screen_size = renderengine.get_instance().get_game_size()
            offs = (-(screen_size[0] // 2 - world_center[0]), -(screen_size[1] // 2 - world_center[1]))
            lay.set_offset(*offs)

    def all_sprites(self):
        for spr in self._terrain_sprites.values():
            yield spr
        for spr in self._entity_sprites.values():
            yield spr


class WorldRenderer3D(WorldRenderer):

    def __init__(self):
        super().__init__()
        self._sprite_3ds = {}  # uid -> list of Sprite3D
        self.camera = threedee.Camera3D(position=(-3.8, 3.2, 2.5), direction=(0.9, -0.4, 0), fov=45)

    def _get_or_make_spr3d(self, spr_id, layer=spriteref.LAYER_3D, idx=0):
        if spr_id in self._sprite_3ds:
            sprs = self._sprite_3ds[spr_id]
            if idx < len(sprs):
                return sprs[idx]
        return threedee.Sprite3D(layer)

    def _all_sprites_for(self, e: Entity, spr_id):
        x, z, y = e.get_xyz()  # 3d coordinates, z and y are swapped intentionally
        edir = e.get_direction()
        rot = -math.atan2(edir[1], edir[0]) + math.pi / 2

        if isinstance(e, Forklift):
            forklift_spr = self._get_or_make_spr3d(spr_id)

            fork_pos = (0, e.get_fork_xyz(absolute=False)[2] / 8, 0)
            forklift_spr = forklift_spr.update(new_model=spriteref.ThreeDeeModels.FORKLIFT,
                                               new_position=(x + 0.5, y / 8 + 0.001, z + 0.5),
                                               new_rotation=(0, rot, 0)
                                               ).update_mesh("fork", new_pos=fork_pos)
            yield forklift_spr
        elif isinstance(e, Block):
            # TODO hacks here covering up some issue w/ direction indices & models' base directions
            bb = e.copy().rotate(-e.get_direction_idx() + 1).get_bounding_box()
            spr = self._get_or_make_spr3d(spr_id)
            spr = spr.update(new_model=spriteref.ThreeDeeModels.CUBE,
                             new_position=(bb[0] + bb[3] / 2, bb[2] / 8, bb[1] + bb[4] / 2),
                             new_rotation=(0, rot, 0),
                             new_scale=(bb[3], bb[5] / 8, bb[4]),
                             new_color=colorutils.to_float(e.get_color()))
            yield spr

            if isinstance(e, WinBlock):
                diam_spr = self._get_or_make_spr3d(spr_id, idx=1)
                t = globaltimer.get_elapsed_time() / 1000
                bob_y_offs = 6/8 + 0.1 * math.cos(t * 2 * math.pi)
                rot_offs = t * 0.1 * 2 * math.pi
                diam_spr = diam_spr.update(new_model=spriteref.ThreeDeeModels.DIAMOND,
                                           new_rotation=(0, rot + rot_offs, 0),
                                           new_position=(bb[0] + bb[3] / 2, (bb[2] + bb[5]) / 8 + bob_y_offs, bb[1] + bb[4] / 2),
                                           new_scale=(0.25, 0.333, 0.25),
                                           new_color=colorutils.to_float(e.get_color()))
                yield diam_spr

    def update(self, w: World):
        super().update(w)
        new_sprites = {}

        for t_xy in w.terrain:
            x, y = t_xy
            z = w.terrain[t_xy]
            t_id = f"terr_{(x, y)}"
            if t_id not in self._sprite_3ds:
                spr = threedee.Sprite3D(spriteref.LAYER_3D, spriteref.ThreeDeeModels.SQUARE)
            else:
                spr = self._sprite_3ds[t_id]

            color = (0.15, 0.15, 0.15)
            if (x + y) % 2 == 0:
                color = colorutils.darker(color)

            spr = spr.update(new_model=spriteref.ThreeDeeModels.SQUARE,
                             new_position=(x + 0.5, z / 8, y + 0.5),
                             new_color=color)
            new_sprites[t_id] = spr

        forklift_spr = None

        for e in w.all_entities():
            old_ent, new_ent, t = self.get_cur_entity(e)
            if (old_ent is new_ent) or t == 0:
                # not interpolating
                sprs = list(x for x in self._all_sprites_for(old_ent, old_ent.uid))
                new_sprites[old_ent.uid] = sprs
            else:
                old_spr_id = (old_ent.uid, 0)
                new_spr_id = (new_ent.uid, 1)
                old_sprs = list(x for x in self._all_sprites_for(old_ent, old_spr_id))
                new_sprs = list(x for x in self._all_sprites_for(new_ent, new_spr_id))

                sprs = []
                for old_spr, new_spr in zip(old_sprs, new_sprs):
                    if old_spr is None and new_spr is not None:
                        sprs.append(new_spr)
                    elif old_spr is not None and new_spr is not None:
                        sprs.append(old_spr.interpolate(new_spr, t))
                    else:
                        pass  # sprite is considered non-existant

                new_sprites[old_spr_id] = old_sprs
                new_sprites[new_spr_id] = new_sprs
                new_sprites[e.uid] = sprs
            if isinstance(e, Forklift) and len(sprs) > 0:
                forklift_spr = sprs[0]

        self._sprite_3ds = new_sprites
        for lay in renderengine.get_instance().get_layers(spriteref.world_3d_layer_ids()):
            if forklift_spr is not None:
                lay.set_light_sources([(util.add(forklift_spr.position(), (0, 2, 0)),
                                        colorutils.lighter(forklift_spr.color(), 0.7))])
        self.update_camera()

    def update_camera(self, new_camera=None):
        if new_camera is not None:
            self.camera = new_camera
        for lay in renderengine.get_instance().get_layers(spriteref.world_3d_layer_ids()):
            lay.set_camera(self.camera)

    def get_sprite_for_ent(self, ent_or_id) -> threedee.Sprite3D:
        if ent_or_id is None:
            return None
        if isinstance(ent_or_id, Entity):
            ent_or_id = ent_or_id.uid
        if ent_or_id in self._sprite_3ds:
            res = self._sprite_3ds[ent_or_id]
            if isinstance(res, list):
                if len(res) > 0:
                    return res[0]
                else:
                    return None
            else:
                return res
        return None

    def all_sprites(self):
        for key in self._sprite_3ds:
            if not isinstance(key, tuple):  # tuples are the before & after sprites
                res = self._sprite_3ds[key]
                if isinstance(res, sprites.AbstractSprite):
                    yield res
                else:
                    for spr in res:
                        yield spr


if __name__ == "__main__":
    ent = Entity((0, 0, 0), [(0, 0, 0), (1, 0, 0)])
    print(ent)
    print(ent.rotate(1))
    print(ent.collides(ent.copy().rotate(1)))
