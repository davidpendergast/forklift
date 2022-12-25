import pygame

import src.engine.game as game
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.scenes as scenes
import src.engine.sprites as sprites
import src.engine.layers as layers

import src.game.spriteref as spriteref
import src.game.menus as menus


class ForkliftGame(game.Game):

    def get_sheets(self):
        return spriteref.initialize_sheets()

    def get_layers(self):
        yield threedee.ThreeDeeLayer(spriteref.LAYER_3D, 1)
        yield layers.ImageLayer(spriteref.LAYER_DEBUG, 1000)

    def initialize(self):
        globaltimer.set_show_fps(True)
        scenes.create_instance(menus.Test3DMenu(spriteref.ThreeDeeModels.FORKLIFT))

    def update(self) -> bool:
        scenes.get_instance().update()
        return True

    def all_sprites(self):
        return scenes.get_instance().all_sprites()