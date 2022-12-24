import pygame

import src.utils.util as util

import src.engine.scenes as scenes
import src.engine.inputs as inputs
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.renderengine as renderengine

import src.game.spriteref as spriteref


class Test3DMenu(scenes.Scene):

    def __init__(self, models):
        super().__init__()

        self.sprites = [threedee.Sprite3D(m, spriteref.LAYER_3D) for m in util.listify(models)]
        self.camera = threedee.KeyboardControlledCamera3D((-20, 2, 0), (1, 0, 0))

        self.lock_cam_to_model = False
        self.perspective_cam = True

    def all_sprites(self):
        return self.sprites

    def update(self):
        if inputs.get_instance().was_pressed(pygame.K_l):
            self.lock_cam_to_model = not self.lock_cam_to_model
        if inputs.get_instance().was_pressed(pygame.K_o):
            self.perspective_cam = not self.perspective_cam
            renderengine.get_instance().get_layer(spriteref.LAYER_3D).set_use_perspective(self.perspective_cam)

        self.camera.update(globaltimer.dt())

        if self.lock_cam_to_model and len(self.sprites) > 0:
            model_pos = self.sprites[0].position()
            cam_pos = self.camera.get_position()
            view_dir = util.set_length(util.sub(model_pos, cam_pos), 1)
            self.camera.set_direction(view_dir)

        renderengine.get_instance().get_layer(spriteref.LAYER_3D).set_camera(self.camera)
