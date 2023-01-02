import math
import random

import pygame

import configs
import src.utils.util as util
import src.utils.colorutils as colorutils

import src.engine.scenes as scenes
import src.engine.inputs as inputs
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.renderengine as renderengine
import src.engine.sprites as sprites
import src.engine.spritesheets as spritesheets

import src.game.spriteref as spriteref
import src.game.world as world


def _build_demo_sprites():
    w = world.build_sample_world()

    res = []

    sc = 4
    for t_xy in w.terrain:
        x, y = t_xy
        z = w.terrain[t_xy]
        res.append(threedee.Sprite3D(spriteref.ThreeDeeModels.SQUARE, spriteref.LAYER_3D,
                                     (x, z, y), color=(0.15, 0.15, 0.15), scale=(sc, sc, sc)))  # coordinates man

    sc = 2
    for e in w.all_entities():
        if isinstance(e, world.Forklift):  # need the forklift to be first for lock-on
            x, z, y = e.xyz
            res.append(threedee.Sprite3D(spriteref.ThreeDeeModels.FORKLIFT_STATIC, spriteref.LAYER_3D,
                                         position=(x + 0.5, y / 8 + 0.001, z + 0.5),
                                         scale=(0.1, 0.1, 0.1),
                                         rotation=(0, 0, 0)))
        if isinstance(e, world.Block):
            bb = e.get_bounding_box()
            res.append(threedee.Sprite3D(spriteref.ThreeDeeModels.CUBE, spriteref.LAYER_3D,
                                         position=(bb[0], bb[2], bb[1]), scale=(sc * bb[3], sc * bb[5] / 8, sc * bb[4]),
                                         color=(colorutils.to_float(e.get_debug_color()))))

    return res


class Test3DMenu(scenes.Scene):

    def __init__(self, models="demo"):
        super().__init__()

        if models == 'demo':
            self.sprites = _build_demo_sprites()
        else:
            self.sprites = [threedee.Sprite3D(m, spriteref.LAYER_3D) for m in util.listify(models)]
        self.camera = threedee.KeyboardControlledCamera3D((-3, 1, 1.5), (1, 0, 0))

        self.lock_cam_to_model = False
        self.enable_lighting = True

        self.flift_pos = self.get_forklift().position()
        self.flift_xz_dir = (0, 1)

        self.debug_info = {}
        self.debug_info_sprite = None

    def get_forklift(self) -> threedee.Sprite3D:
        for spr in self.sprites:
            if spr.model() == spriteref.ThreeDeeModels.FORKLIFT_STATIC:
                return spr
        return None

    def set_forkflit(self, flift):
        idx = self.sprites.index(self.get_forklift())
        if idx >= 0:
            self.sprites[idx] = flift
        else:
            self.sprites.append(flift)

    def update_forklift(self, dt):
        flift = self.get_forklift()
        if flift is None:
            return None

        move_speed = 0.4  # units per sec
        turn_speed = 45  # deg per sec
        new_xz_dir = util.rotate(self.flift_xz_dir, dt / 1000 * turn_speed * (math.pi / 180))
        new_pos = (self.flift_pos[0] + util.set_length(new_xz_dir, dt / 1000 * move_speed)[0],
                   self.flift_pos[1],
                   self.flift_pos[2] + util.set_length(new_xz_dir, dt / 1000 * move_speed)[1])

        self.flift_pos = new_pos
        self.flift_xz_dir = new_xz_dir

        new_flift = flift.update(new_position=self.flift_pos,
                                 new_rotation=(0, -math.atan2(self.flift_xz_dir[1], self.flift_xz_dir[0]) + math.pi / 2, 0))
        self.set_forkflit(new_flift)
        return new_flift

    def all_sprites(self):
        for spr in self.sprites:
            yield spr
        yield self.debug_info_sprite

    def update_debug_info_sprite(self):
        self.debug_info["position"] = self.camera.get_position()

        direction = self.camera.get_direction()
        self.debug_info["direction (c)"] = direction
        self.debug_info["direction (s)"] = pygame.Vector3(direction[0], direction[2], direction[1]).as_spherical()

        self.debug_info["fov"] = self.camera.get_fov()

        text = sprites.TextBuilder()
        for key in self.debug_info:
            val = self.debug_info[key]
            val_str = str(val)
            if isinstance(val, (pygame.Vector2, pygame.Vector3)):
                val = tuple(v for v in val)
            if isinstance(val, tuple):
                val_str = f"({', '.join('{:.3f}'.format(v) for v in val)})"
            elif isinstance(val, (float, int)):
                val_str = '{:.3f}'.format(val)
            text.addLine(f"{key:16}:{val_str}")

        if self.debug_info_sprite is None:
            self.debug_info_sprite = sprites.TextSprite(spriteref.LAYER_DEBUG, 0, 0, "abc",
                                                        font_lookup=spritesheets.get_default_font(True, False))
        self.debug_info_sprite.update(new_text=text)

    def update(self):
        if inputs.get_instance().was_pressed(configs.DEBUG_TOGGLE_FREE_CAMERA):
            self.lock_cam_to_model = not self.lock_cam_to_model

        layer3d = renderengine.get_instance().get_layer(spriteref.LAYER_3D)
        if inputs.get_instance().was_pressed(configs.DEBUG_TOGGLE_ORTHO_CAMERA):
            layer3d.set_use_perspective(not layer3d.use_perspective)
        if inputs.get_instance().was_pressed(configs.DEBUG_TOGGLE_TEXTURES):
            layer3d.set_show_textures(not layer3d.show_textures)
        if inputs.get_instance().was_pressed(configs.DEBUG_TOGGLE_WIREFRAME):
            layer3d.set_show_wireframe(not layer3d.show_wireframe)
        if inputs.get_instance().was_pressed(configs.DEBUG_TOGGLE_LIGHTING):
            layer3d.set_use_lighting(not layer3d.use_lighting)

        self.camera.update()

        forklift_spr = self.update_forklift(globaltimer.dt())
        if forklift_spr is not None:
            layer3d.set_light_sources([(util.add(forklift_spr.position(), (0, 2, 0)),
                                        colorutils.lighter(forklift_spr.color(), 0.7))])

            if self.lock_cam_to_model:
                model_pos = forklift_spr.position()
                cam_pos = self.camera.get_position()
                view_dir = util.set_length(util.sub(model_pos, cam_pos), 1)
                self.camera.set_direction(view_dir)

        layer3d.set_camera(self.camera)
        self.update_debug_info_sprite()


class InGameScene(scenes.Scene):

    def __init__(self, world_state: world.World):
        super().__init__()
        self.world_state = world_state
        self.renderers = [world.WorldRenderer2D(), world.WorldRenderer3D()]
        self.active_renderer_idx = 0

    def get_renderer(self) -> world.WorldRenderer:
        return self.renderers[self.active_renderer_idx]

    def all_sprites(self):
        for spr in super().all_sprites():
            yield spr
        for spr in self.get_renderer().all_sprites():
            yield spr

    def update(self):
        if inputs.get_instance().was_pressed(pygame.K_F2):
            self.active_renderer_idx = (self.active_renderer_idx + 1) % len(self.renderers)

        flift = self.world_state.get_forklift()
        if flift is not None:
            dxy = inputs.get_instance().was_pressed_four_way(left=configs.MOVE_LEFT, right=configs.MOVE_RIGHT,
                                                             negy=configs.MOVE_UP, posy=configs.MOVE_DOWN)
            if dxy[0] != 0:
                flift = flift.rotate(dxy[0])
                self.world_state.add_entity(flift)

            if dxy[1] != 0:
                xyz = flift.get_xyz()
                move_dir = util.mult(flift.get_direction(), dxy[1])
                next_xy = (xyz[0] + move_dir[0], xyz[1] + move_dir[1])

                next_xyz = (*next_xy, 0)
                for z in range(self.world_state.get_terrain_height(next_xy, or_else=0), 32):
                    next_xyz = (*next_xy, z)
                    avail = True
                    for _ in self.world_state.all_entities_colliding_with(flift, xyz_override=next_xyz):
                        avail = False
                        break
                    if avail:
                        break

                flift = flift.set_xyz(next_xyz)
                self.world_state.add_entity(flift)
                print(f"INFO: forklift moved to ({flift.get_xyz()})")

            df = inputs.get_instance().was_pressed_two_way(negative=configs.CROUCH, positive=configs.JUMP)
            if df > 0:
                world.ForkliftActionHandler.raise_fork(flift, self.world_state)

            if df != 0:
                flift = flift.move_fork(df)
                self.world_state.add_entity(flift)


        self.renderers[self.active_renderer_idx].update(self.world_state)