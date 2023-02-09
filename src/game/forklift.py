import pygame
import os

import src.engine.game as game
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.scenes as scenes
import src.engine.sprites as sprites
import src.engine.layers as layers
import src.engine.readme_writer as readme_writer
import src.engine.keybinds as keybinds
import src.utils.util as util
import configs

import src.game.spriteref as spriteref
import src.game.menus as menus
import src.game.world as world


class ForkliftGame(game.Game):

    def get_sheets(self):
        return spriteref.initialize_sheets()

    def get_layers(self):
        yield threedee.ThreeDeeLayer(spriteref.LAYER_3D, 1)
        yield layers.PolygonLayer(spriteref.LAYER_POLY, 500)
        yield layers.ImageLayer(spriteref.LAYER_WORLD_2D, 750)
        yield layers.ImageLayer(spriteref.LAYER_DEBUG, 1000)

    def initialize(self):
        if configs.is_dev:
            _update_readme()

        kb = keybinds.get_instance()
        kb.set_binding(configs.MOVE_UP, keybinds.Binding(pygame.K_w, mods=(pygame.KMOD_NONE)))
        kb.set_binding(configs.MOVE_LEFT, pygame.K_a)
        kb.set_binding(configs.MOVE_RIGHT, pygame.K_d)
        kb.set_binding(configs.MOVE_DOWN, pygame.K_s)
        kb.set_binding(configs.JUMP, pygame.K_SPACE)
        kb.set_binding(configs.CROUCH, pygame.K_c)

        kb.set_binding(configs.ENTER, pygame.K_RETURN)
        kb.set_binding(configs.ESCAPE, pygame.K_ESCAPE)
        kb.set_binding(configs.RESET, pygame.K_r)

        kb.set_binding(configs.ROTATE_UP, pygame.K_UP)
        kb.set_binding(configs.ROTATE_LEFT, pygame.K_LEFT)
        kb.set_binding(configs.ROTATE_RIGHT, pygame.K_RIGHT)
        kb.set_binding(configs.ROTATE_DOWN, pygame.K_DOWN)

        kb.set_binding(configs.DEBUG_TOGGLE_WIREFRAME, keybinds.Binding(pygame.K_w, mods=(pygame.KMOD_CTRL)))
        kb.set_binding(configs.DEBUG_TOGGLE_TEXTURES, keybinds.Binding(pygame.K_t, mods=(pygame.KMOD_CTRL)))
        kb.set_binding(configs.DEBUG_TOGGLE_RAINBOW, keybinds.Binding(pygame.K_t, mods=(pygame.KMOD_NONE)))
        kb.set_binding(configs.DEBUG_INCREASE_CAMERA_FOV, keybinds.Binding(pygame.K_f, mods=(pygame.KMOD_NONE)))
        kb.set_binding(configs.DEBUG_DECREASE_CAMERA_FOV, keybinds.Binding(pygame.K_f, mods=(pygame.KMOD_SHIFT)))
        kb.set_binding(configs.DEBUG_TOGGLE_FREE_CAMERA, keybinds.Binding(pygame.K_l, mods=(pygame.KMOD_CTRL)))
        kb.set_binding(configs.DEBUG_TOGGLE_ORTHO_CAMERA, pygame.K_o)

        kb.set_binding(configs.DEBUG_TOGGLE_LIGHTING, keybinds.Binding(pygame.K_l, mods=(pygame.KMOD_NONE)))

        globaltimer.set_show_fps(True)

        # scenes.create_instance(menus.Test3DMenu())
        scenes.create_instance(menus.InGameScene(idx=0))

    def update(self) -> bool:
        scenes.get_instance().update()
        return True

    def all_sprites(self):
        return scenes.get_instance().all_sprites()


def _update_readme():
    gif_directory = "gifs"
    gif_filenames = [f for f in os.listdir(gif_directory) if os.path.isfile(os.path.join(gif_directory, f))]
    gif_filenames = [f for f in gif_filenames if f.endswith(".gif") and f[0].isdigit()]
    gif_filenames.sort(key=lambda text: util.parse_leading_int(text, or_else=-1), reverse=True)

    def _key_lookup(key: str):
        n = util.parse_ending_int(key, or_else=-1)
        if n < 0 or n >= len(gif_filenames):
            return None
        if key.startswith("file_"):
            return gif_filenames[n]
        elif key.startswith("name_"):
            return gif_filenames[n][:-4]  # rm the ".gif" part
        else:
            return None

    readme_writer.write_readme("README_template.txt", "README.md",
                               key_lookup=_key_lookup,
                               skip_line_if_value_missing=True)