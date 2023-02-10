import math
import typing

import configs
import src.utils.util as util

import src.engine.threedee as threedee
import src.engine.spritesheets as spritesheets
import src.engine.sprites as sprites


LAYER_DEBUG = "debug"
LAYER_WORLD_2D = "world_2d"
LAYER_POLY = "poly"
LAYER_3D = "threedee"


def world_2d_layer_ids():
    return (LAYER_POLY, LAYER_WORLD_2D)


def world_3d_layer_ids():
    return (LAYER_3D,)


def _init_sheet(sheet_id, path, sheet_list):
    return util.add_to_list(spritesheets.SpriteSheet(sheet_id, util.resource_path(path)), sheet_list)


def _img(x, y, w, h, offs=(0, 0), trans=False, l=None) -> sprites.ImageModel:
    res = sprites.ImageModel(x, y, w, h, offset=offs, translucent=trans)
    if l is not None:
        l.append(res)
    return res


class _CharacterSheet(spritesheets.SpriteSheet):

    def __init__(self):
        super().__init__("character_sheet", "assets/sprites.png")
        self.skeletons = []
        self.ghosts = []
        self.vampires = []
        self.tormented_souls = []
        self.demons = []
        self.maggots = []

    def draw_to_atlas(self, atlas, sheet, start_pos=(0, 0)):
        super().draw_to_atlas(atlas, sheet, start_pos=start_pos)
        self.skeletons = [_img(16 * i, 0, 15, 32) for i in range(2)]
        self.ghosts = [_img(32 + 16 * i, 0, 15, 32) for i in range(2)]
        self.vampires = [_img(64 + 16 * i, 0, 15, 48) for i in range(2)]
        self.tormented_souls = [_img(16 * i, 64, 16 + 15, 32) for i in range(2)]
        self.demons = [_img(64 + 32 * i, 48, 16 + 15, 48) for i in range(2)]
        self.maggots = [_img(0, 32 + i * 16, 16 + 15, 16) for i in range(2)]


CHARACTER_SHEET = _CharacterSheet()


_3D_TEXTURES = {}  # sheet_id -> TextureSheet


class ThreeDeeTextureSheets:
    ALL_SHEETS = []
    FORKLIFT = _init_sheet("forklift", "assets/models/forklift_default.png", ALL_SHEETS)
    SKYBOX_GREEN = _init_sheet("skybox", "assets/models/skybox.png", ALL_SHEETS)


class ThreeDeeModels:
    FORKLIFT_STATIC = None
    FORKLIFT = None
    SQUARE = None
    VERT_SQUARE = None
    CUBE = None
    SPHERE = None
    WEDGE = None
    CYLINDER = None
    DIAMOND = None
    SKYBOX = None

    _2D_MODEL_CACHE = {}  # ImageModel -> ThreeDeeModel

    @staticmethod
    def load_models_from_disk():
        TDMM = threedee.ThreeDeeMultiMesh
        TDM = threedee.ThreeDeeMesh

        ThreeDeeModels.FORKLIFT = TDMM.load_from_disk(
            "forklift", util.resource_path("assets/models/forklift/"),
            {
                "body": "body.obj",
                "fork": "fork.obj",
                "wheels": "wheels.obj"
            }
        ).apply_scaling((0.15, 0.15, 0.15))\
            .apply_translation((0, 0, 0.45))

        ThreeDeeModels.FORKLIFT_STATIC = TDM.load_from_disk(
            "forklift_static",
            util.resource_path("assets/models/forklift.obj")
        )

        # model is 1x0x1, origin is at center
        ThreeDeeModels.SQUARE = TDM.load_from_disk("square", util.resource_path("assets/models/square.obj"))\
            .apply_scaling((4, 4, 4))\
            .apply_translation((-0.5, 0, -0.5))

        # model is 1x1x0, origin is at bottom center
        ThreeDeeModels.VERT_SQUARE = TDM.load_from_disk("square", util.resource_path("assets/models/square.obj")) \
            .apply_scaling((4, 4, 4)) \
            .apply_rotation((-math.pi / 2, 0, 0)) \
            .apply_translation((-0.5, 1, 0))

        # model is 1x1x1, origin is at center of bottom face
        ThreeDeeModels.CUBE = TDM.load_from_disk("cube", util.resource_path("assets/models/cube.obj"))\
            .apply_scaling((4, 4, 4))\
            .apply_translation((-0.5, 0, -0.5))

        # model is 1x1x1, origin is at the center
        ThreeDeeModels.SPHERE = TDM.load_from_disk("sphere", util.resource_path("assets/models/sphere8.obj"))

        # model is 1x1x1, origin is at the center of bottom face, "wedge" goes upward in the positive y direction:
        #  / |  --> y
        # /__|
        ThreeDeeModels.WEDGE = TDM.load_from_disk("wedge", util.resource_path("assets/models/wedge.obj"))

        # model is 1x1x1, origin is at the center of the bottom face
        ThreeDeeModels.CYLINDER = TDM.load_from_disk("cylinder", util.resource_path("assets/models/cylinder.obj"))

        # model is 1x1x1-ish, origin is at the center
        ThreeDeeModels.DIAMOND = TDM.load_from_disk("diamond", util.resource_path("assets/models/diamond.obj"))

        # model is 1x1x1, origin is at center
        ThreeDeeModels.SKYBOX = TDM.load_from_disk("skybox", util.resource_path("assets/models/inverted_textured_cube.obj")) \
            .apply_scaling((4, 4, 4)) \
            .apply_translation((-0.5, -0.5, -0.5))

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

        if configs.debug_corners:
            x, y, w, h = start_pos[0], start_pos[1], sheet.get_width(), sheet.get_height()
            atlas.set_at((x, y), (0, 0, 0))
            atlas.set_at((x + w - 1, y), (255, 0, 0))
            atlas.set_at((x + w - 1, y + h - 1), (255, 0, 255))
            atlas.set_at((x, y + h - 1), (0, 0, 255))

        def _map_to_atlas(xy):
            atlas_x = (sheet_rect[0] + xy[0] * sheet_rect[2])
            atlas_y = atlas_size[1] - (sheet_rect[1] + (1 - xy[1]) * sheet_rect[3])
            return (atlas_x, atlas_y)

        self._texture_coord_to_atlas_coord = _map_to_atlas


def initialize_sheets() -> typing.List[spritesheets.SpriteSheet]:
    all_sheets = []
    all_sheets.extend(ThreeDeeTextureSheets.ALL_SHEETS)
    all_sheets.append(CHARACTER_SHEET)

    ThreeDeeModels.load_models_from_disk()

    return all_sheets


