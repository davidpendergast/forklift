import typing

import configs
import src.utils.util as util

import src.engine.threedee as threedee
import src.engine.spritesheets as spritesheets


LAYER_3D = "threedee"
LAYER_DEBUG = "debug"
LAYER_POLY = "poly"
LAYER_WORLD_2D = "world_2d"


def world_2d_layer_ids():
    return (LAYER_POLY, LAYER_WORLD_2D)


def world_3d_layer_ids():
    return (LAYER_3D,)


_3D_TEXTURES = {}  # sheet_id -> TextureSheet


class TextureSheetTypes:
    ALL_TYPES = []
    RAINBOW = util.add_to_list(("rainbow", "assets/models/rainbow.png"), ALL_TYPES)
    WHITE = util.add_to_list(("white", "assets/models/white.png"), ALL_TYPES)
    FORKLIFT_DEFAULT = util.add_to_list(("forklift_default", "assets/models/forklift_default.png"), ALL_TYPES)


class ThreeDeeModels:
    FORKLIFT_STATIC = None
    FORKLIFT = None
    SQUARE = None
    CUBE = None

    _2D_MODEL_CACHE = {}  # ImageModel -> ThreeDeeModel

    @staticmethod
    def _get_xform_for_texture(texture_id):
        if configs.rainbow_3d:
            texture_id = "rainbow"
        return lambda xy: _3D_TEXTURES[texture_id].get_xform_to_atlas()(xy)

    @staticmethod
    def load_models_from_disk():
        xform = lambda ident: ThreeDeeModels._get_xform_for_texture(ident)
        s = lambda path: util.resource_path(path)

        ThreeDeeModels.FORKLIFT = threedee.ThreeDeeMultiMesh.load_from_disk("forklift", s("assets/models/forklift/"),
                                                                            {"body": "body.obj",
                                                                             "fork": "fork.obj",
                                                                             "wheels": "wheels.obj"
                                                                             }, xform("forklift_default"))
        ThreeDeeModels.FORKLIFT_STATIC = threedee.ThreeDeeMesh.load_from_disk("forklift_static",
                                                                              s("assets/models/forklift.obj"),
                                                                              xform("forklift_default"))

        ThreeDeeModels.SQUARE = threedee.ThreeDeeMesh.load_from_disk("square", s("assets/models/square.obj"),
                                                                     xform("white"))
        ThreeDeeModels.CUBE = threedee.ThreeDeeMesh.load_from_disk("cube", s("assets/models/cube.obj"),
                                                                   xform("white"))

    @staticmethod
    def from_2d_model(model_2d):
        if model_2d not in ThreeDeeModels._2D_MODEL_CACHE:
            ThreeDeeModels._2D_MODEL_CACHE[model_2d] = threedee.ThreeDeeMesh.build_from_2d_model(model_2d)
        return ThreeDeeModels._2D_MODEL_CACHE[model_2d]


class TextureSheet(spritesheets.SpriteSheet):

    def __init__(self, sheet_id, filename):
        spritesheets.SpriteSheet.__init__(self, sheet_id, filename)

        self._texture_coord_to_atlas_coord = lambda xy: None

    def get_xform_to_atlas(self):
        return self._texture_coord_to_atlas_coord

    def draw_to_atlas(self, atlas, sheet, start_pos=(0, 0)):
        super().draw_to_atlas(atlas, sheet, start_pos=start_pos)
        atlas_size = (atlas.get_width(), atlas.get_height())
        sheet_rect = [start_pos[0], start_pos[1], sheet.get_width(), sheet.get_height()]

        def _map_to_atlas(xy):
            atlas_x = (sheet_rect[0] + xy[0] * sheet_rect[2])
            atlas_y = atlas_size[1] - (sheet_rect[1] + (1 - xy[1]) * sheet_rect[3])
            return (atlas_x, atlas_y)

        self._texture_coord_to_atlas_coord = _map_to_atlas


def initialize_sheets() -> typing.List[spritesheets.SpriteSheet]:
    all_sheets = []

    for id_and_file in TextureSheetTypes.ALL_TYPES:
        sheet_id, filepath = id_and_file
        _3D_TEXTURES[sheet_id] = TextureSheet(sheet_id, filepath)
        all_sheets.append(_3D_TEXTURES[sheet_id])

    ThreeDeeModels.load_models_from_disk()

    return all_sheets


