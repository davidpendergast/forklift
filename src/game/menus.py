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

    for e in w.all_entities():
        if isinstance(e, world.Forklift):  # need the forklift to be first for lock-on
            x, z, y = e.xyz
            flift = threedee.Sprite3D(spriteref.ThreeDeeModels.FORKLIFT, spriteref.LAYER_3D,
                                      position=(x + 0.5, y / 8 + 0.001, z + 0.5),
                                      scale=(0.1, 0.1, 0.1),
                                      rotation=(0, 0, 0))
            flift.color = lambda: colorutils.rainbow(globaltimer.get_elapsed_time(), 1000, s=0.5)
            res.append(flift)

    sc = 4
    for t_xy in w.terrain:
        x, y = t_xy
        z = w.terrain[t_xy]
        res.append(threedee.Sprite3D(spriteref.ThreeDeeModels.SQUARE, spriteref.LAYER_3D,
                                     (x, z, y), color=(0.15, 0.15, 0.15), scale=(sc, sc, sc)))  # coordinates man

    sc = 2
    for e in w.all_entities():
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
        self.camera = threedee.KeyboardControlledCamera3D((-20, 2, 0), (1, 0, 0))

        self.lock_cam_to_model = False

        self.debug_info = {}
        self.debug_info_sprite = None

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

        self.camera.update()

        if self.lock_cam_to_model and len(self.sprites) > 0:
            model_pos = self.sprites[0].position()
            cam_pos = self.camera.get_position()
            view_dir = util.set_length(util.sub(model_pos, cam_pos), 1)
            self.camera.set_direction(view_dir)

        renderengine.get_instance().get_layer(spriteref.LAYER_3D).set_camera(self.camera)

        self.update_debug_info_sprite()
