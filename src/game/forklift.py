import pygame
import os

import src.engine.game as game
import src.engine.threedee as threedee
import src.engine.globaltimer as globaltimer
import src.engine.scenes as scenes
import src.engine.sprites as sprites
import src.engine.layers as layers
import src.engine.readme_writer as readme_writer
import src.utils.util as util
import configs

import src.game.spriteref as spriteref
import src.game.menus as menus


class ForkliftGame(game.Game):

    def get_sheets(self):
        return spriteref.initialize_sheets()

    def get_layers(self):
        yield threedee.ThreeDeeLayer(spriteref.LAYER_3D, 1)
        yield layers.ImageLayer(spriteref.LAYER_DEBUG, 1000)

    def initialize(self):
        if configs.is_dev:
            _update_readme()

        globaltimer.set_show_fps(True)
        scenes.create_instance(menus.Test3DMenu())

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