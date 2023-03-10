import traceback
import datetime
import os
import pathlib

import configs
import src.game.forklift


"""
The main entry point.
"""

game_class = src.game.forklift.ForkliftGame


def _dismiss_splash_screen():
    try:
        import pyi_splash  # special pyinstaller thing - import will not resolve in dev
        pyi_splash.close()
    except Exception:
        pass  # this is expected to throw an exception in non-splash launch contexts.


if __name__ == "__main__":
    _dismiss_splash_screen()

    try:
        import src.engine.gameloop as gameloop
        loop = gameloop.create_instance(game_class())
        loop.run()

    except Exception as e:
        if configs.do_crash_reporting:
            import src.engine.crashreporting as crashreporting
            crashreporting.write_crash_file(configs.name_of_game, configs.version, dest_dir="logs")

        raise e

