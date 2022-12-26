import pygame

import src.utils.util as util

import src.engine.scenes as scenes
import src.engine.inputs as inputs
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.renderengine as renderengine
import src.engine.sprites as sprites
import src.engine.spritesheets as spritesheets

import src.game.spriteref as spriteref


class Test3DMenu(scenes.Scene):

    def __init__(self, models):
        super().__init__()

        self.sprites = [threedee.Sprite3D(m, spriteref.LAYER_3D) for m in util.listify(models)]
        self.camera = threedee.KeyboardControlledCamera3D((-20, 2, 0), (1, 0, 0), fov=25)

        self.lock_cam_to_model = False
        self.perspective_cam = True

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
        if inputs.get_instance().was_pressed(pygame.K_l):
            self.lock_cam_to_model = not self.lock_cam_to_model
        if inputs.get_instance().was_pressed(pygame.K_o):
            self.perspective_cam = not self.perspective_cam
            renderengine.get_instance().get_layer(spriteref.LAYER_3D).set_use_perspective(self.perspective_cam)

        self.camera.update()

        if self.lock_cam_to_model and len(self.sprites) > 0:
            model_pos = self.sprites[0].position()
            cam_pos = self.camera.get_position()
            view_dir = util.set_length(util.sub(model_pos, cam_pos), 1)
            self.camera.set_direction(view_dir)

        renderengine.get_instance().get_layer(spriteref.LAYER_3D).set_camera(self.camera)

        self.update_debug_info_sprite()
