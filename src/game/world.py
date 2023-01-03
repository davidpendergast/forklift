import math
import typing

import src.utils.util as util
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

    def get_xyz(self):
        return self.xyz

    def set_xyz(self, xyz) -> 'Entity':
        self.xyz = xyz
        return self

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

    def get_debug_color(self):
        return (55, 55, 55)

    def copy(self, keep_uid=True) -> 'Entity':
        if type(self) is Entity:
            return Entity(self.xyz, self.cells, uid=self.uid if keep_uid else None)
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

    def copy(self, keep_uid=True) -> 'Block':
        return Block(self.get_xyz(), self.get_cells(absolute=False), self.color, self.liftable,
                     uid=self.uid if keep_uid else None)


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Forklift(Entity):

    def __init__(self, xyz, direction=DIRECTIONS[0], fork_z=0, fork_z_bounds=(0, 8), uid=None):
        super().__init__(xyz, [(0, 0, z) for z in range(0, 6)], uid=uid)
        self.direction = direction
        self.fork_z_bounds = fork_z_bounds
        self.fork_z = util.bound(fork_z, *fork_z_bounds)

    def get_debug_color(self):
        return (224, 224, 64)

    def get_direction(self):
        return self.direction

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

    def rotate(self, cw_cnt=1, rel_pivot_pt=None) -> 'Forklift':
        super().rotate(cw_cnt=1, rel_pivot_pt=None)
        dir_idx = DIRECTIONS.index(self.get_direction())
        self.direction = DIRECTIONS[(dir_idx + cw_cnt) % 4]
        return self

    def copy(self, keep_uid=True) -> 'Forklift':
        return Forklift(self.get_xyz(), self.get_direction(),
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
            if log: print(f"INFO: fork is already at max height ({forklift.fork_z})")
            return None
        else:
            stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state,
                                                                  cond=lambda e: e.is_liftable())
            # check to see if the stack is blocked from above
            mut = WorldMutation()
            mut.updates[forklift] = forklift.copy().move_fork(1, safe=False)
            for stack_ent in stack_on_fork.entities:
                if stack_ent == forklift:
                    if log: print("INFO: you can't lift yourself ._.")
                    return None

                for col_ent in world_state.all_entities_colliding_with(stack_ent, offs=(0, 0, 1),
                                                                       cond=lambda e: e not in stack_on_fork):
                    if log: print(f"INFO: stack will collide if lifted ({stack_ent} -> {col_ent})")
                    return None

                mut.updates[stack_ent] = stack_ent.copy().move((0, 0, 1))

            return mut

    @staticmethod
    def lower_fork(forklift: Forklift, world_state: 'World', log=True) -> typing.Optional['WorldMutation']:
        if forklift.fork_z <= forklift.fork_z_bounds[0]:
            if log: print(f"INFO: fork is already at min height ({forklift.fork_z})")
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
            mut.updates[forklift] = forklift.copy().move_fork(-1, safe=False)

            # finally lower the unblocked ents
            for stack_ent in stack_on_fork:
                if stack_ent not in ents_blocked:
                    mut.updates[stack_ent] = stack_ent.copy().move((0, 0, -1))

            return mut

    @staticmethod
    def rotate_forklift(forklift: Forklift, cw_steps, world_state: 'World', all_or_none=True, log=True) \
            -> typing.Optional['WorldMutation']:
        pass

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
        stack_on_fork = ForkliftActionHandler.get_stack_above(forklift.get_fork_xyz(), world_state)
        move_xyz = (*forklift.get_direction(), 0) if forward else (*util.negate(forklift.get_direction()), 0)
        new_forklift = forklift.copy().move(move_xyz)

        # ensure the forklift can move there, ignoring stack
        if not ForkliftActionHandler.is_valid_position_for_forklift(new_forklift, world_state, stack_on_fork=stack_on_fork, log=log):
            return None

        mut = WorldMutation()
        mut.updates[forklift] = new_forklift

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
        moved_ents.add(mut.updates[forklift])

        # move things that aren't blocked
        for moved_ent in stack_on_fork:
            if moved_ent not in blocked_stack_ents:
                mut.updates[moved_ent] = moved_ent.copy().move(move_xyz)
                moved_ents.add(mut.updates[moved_ent])

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
                    if world_state.get_terrain_height(cell_below[:2], or_else=-float('inf')) == cell_below[2] + 1:
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

    def apply(self, world_state: 'World'):
        raise NotImplementedError()

    def undo(self, world_state: 'World'):
        raise NotImplementedError()


class WorldMutation(AbstractWorldMutation):

    def __init__(self):
        self.updates = {}   # old copy of Entity -> new copy of Entity
        self.additions = {}  # new Entities

    def apply(self, world_state: 'World'):
        for old_ent in self.updates:
            new_ent = self.updates[old_ent]
            world_state.remove_entity(old_ent)
            if new_ent is not None:
                world_state.add_entity(new_ent)
        for ent in self.additions:
            world_state.add_entity(ent)

    def undo(self, world_state: 'World'):
        for ent in self.additions:
            world_state.remove_entity(ent)
        for (old_ent, new_ent) in self.updates.items():
            if new_ent is not None:
                world_state.remove_entity(new_ent)
            world_state.add_entity(old_ent)


class CompositeWorldMutation(AbstractWorldMutation):

    def __init__(self, muts=()):
        self.muts: typing.List[AbstractWorldMutation] = list(muts)

    def apply(self, world_state: 'World'):
        for m in self.muts:
            m.apply(world_state)

    def undo(self, world_state: 'World'):
        for m in reversed(self.muts):
            m.undo(world_state)


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

    def get_terrain_height(self, xy, or_else=-1) -> int:
        if xy in self.terrain:
            return self.terrain[xy]
        else:
            return or_else


def build_sample_world():
    w = World()
    flift = (3, 1), (1, 0)
    holes = {(5, 0), (5, 1), (6, 1), (7, 1), (6, 2), (7, 2), (6, 3), (7, 3)}
    boxes = [(1, 0), (2, 0), (4, 0), (0, 3)]
    planks = [[(0, 0 + i) for i in range(3)],
              [(0 + i, 4) for i in range(3)],
              [(3 + i, 3) for i in range(5)],
              [(0 + i, 0, 8) for i in range(3)]]
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

    def update(self, world_state: World):
        raise NotImplementedError()

    def all_sprites(self):
        raise NotImplementedError()


class WorldRenderer2D(WorldRenderer):

    CELL_SIZE = 64

    def __init__(self):
        super().__init__()
        self._terrain_sprites = {}  # (x, y) -> AbstractSprite
        self._entity_sprites = {}  # ent_id -> AbstractSprite

    def update(self, world_state: World):
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
            color = colorutils.to_float(*ent.get_debug_color())
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
        self.lock_camera_to_forklift = False
        self._sprite_3ds = {}  # uid -> Sprite3D
        self.camera = threedee.Camera3D(position=(-3.8, 3.2, 2.5), direction=(0.9, -0.4, 0), fov=45)

    def update(self, w: World):
        new_sprites = {}

        sc = 4
        for t_xy in w.terrain:
            x, y = t_xy
            z = w.terrain[t_xy]
            t_id = f"terr_{(x, y)}"
            if t_id not in self._sprite_3ds:
                spr = threedee.Sprite3D(spriteref.ThreeDeeModels.SQUARE, spriteref.LAYER_3D)
            else:
                spr = self._sprite_3ds[t_id]

            color = (0.15, 0.15, 0.15)
            if (x + y) % 2 == 0:
                color = colorutils.darker(color)

            spr = spr.update(new_model=spriteref.ThreeDeeModels.SQUARE, new_position=(x, z / 8, y),
                             new_color=color, new_scale=(sc, sc, sc))
            new_sprites[t_id] = spr

        forklift_spr = None

        for e in w.all_entities():
            if isinstance(e, Forklift):  # need the forklift to be first for lock-on
                x, z, y = e.xyz
                if e.uid in self._sprite_3ds:
                    forklift_spr = self._sprite_3ds[e.uid]
                else:
                    forklift_spr = threedee.Sprite3D(spriteref.ThreeDeeModels.FORKLIFT, spriteref.LAYER_3D)
                fdir = e.get_direction()
                rot = -math.atan2(fdir[1], fdir[0]) + math.pi / 2
                forklift_spr = forklift_spr.update(new_model=spriteref.ThreeDeeModels.FORKLIFT,
                                                   new_position=(x + 0.5 + 0.45 * fdir[0], y / 8 + 0.001, z + 0.5 + 0.45 * fdir[1]),
                                                   new_scale=(0.15, 0.15, 0.15),
                                                   new_rotation=(0, rot, 0)
                                                   ).update_mesh("fork", new_pos=(0, e.get_fork_xyz(absolute=False)[2] / 8, 0))
                new_sprites[e.uid] = forklift_spr
            elif isinstance(e, Block):
                bb = e.get_bounding_box()
                if e.uid in self._sprite_3ds:
                    spr = self._sprite_3ds[e.uid]
                else:
                    spr = threedee.Sprite3D(spriteref.ThreeDeeModels.CUBE, spriteref.LAYER_3D)
                spr = spr.update(new_model=spriteref.ThreeDeeModels.CUBE,
                                 new_position=(bb[0], bb[2] / 8, bb[1]),
                                 new_scale=(sc * bb[3], sc * bb[5] / 8, sc * bb[4]),
                                 new_color=(colorutils.to_float(e.get_debug_color())))
                new_sprites[e.uid] = spr

        self._sprite_3ds = new_sprites
        for lay in renderengine.get_instance().get_layers(spriteref.world_3d_layer_ids()):
            lay.set_camera(self.camera)

            if forklift_spr is not None:
                lay.set_light_sources([(util.add(forklift_spr.position(), (0, 2, 0)),
                                        colorutils.lighter(forklift_spr.color(), 0.7))])

                if self.lock_camera_to_forklift:
                    model_pos = forklift_spr.position()
                    cam_pos = self.camera.get_position()
                    view_dir = util.set_length(util.sub(model_pos, cam_pos), 1)
                    self.camera.set_direction(view_dir)

    def all_sprites(self):
        for spr in self._sprite_3ds.values():
            yield spr


if __name__ == "__main__":
    ent = Entity((0, 0, 0), [(0, 0, 0), (1, 0, 0)])
    print(ent)
    print(ent.rotate(1))
    print(ent.collides(ent.copy().rotate(1)))
