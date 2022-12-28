from OpenGL.GL import *

import numpy
import pygame
import math

import src.engine.layers as layers
import src.engine.sprites as sprites
import src.engine.inputs as inputs
import src.utils.util as util
import src.utils.matutils as matutils
import src.engine.globaltimer as globaltimer
import src.engine.renderengine as renderengine
import configs


class Camera3D:

    def __init__(self, position=(0, 0, 0), direction=(0, 0, -1), fov=45):
        self._position = position
        self._direction = direction
        self._fov = fov

    def get_position(self):
        return self._position

    def set_position(self, position):
        self._position = position

    def get_direction(self):
        return self._direction

    def get_up_vector(self):
        return (0, 1, 0)

    def get_snapshot(self) -> 'Camera3D':
        """returns a copy of this camera"""
        return Camera3D(position=self.get_position(),
                        direction=self.get_direction(),
                        fov=self.get_fov())

    def set_direction(self, direction):
        self._direction = direction

    def get_fov(self):
        return self._fov

    def set_fov(self, fov):
        self._fov = fov

    def update(self):
        pass


_ADJUST = 1/5


class KeyboardControlledCamera3D(Camera3D):

    def __init__(self, position=(0, 0, 0), direction=(0, 0, -1), fov=45,
                 move_speed=(30*_ADJUST, 30*_ADJUST),
                 rot_speed=(60, 45)):
        super().__init__(position=position, direction=direction, fov=fov)
        self.hrot_speed = rot_speed[0]
        self.vrot_speed = rot_speed[1]
        self.move_speed = move_speed[0]
        self.fly_speed = move_speed[1]
        self.v_angle_limit = (10, 170)

        self._mouse_down_xy = None
        self._mouse_down_direction = None

    def update(self):
        dt_secs = globaltimer.dt() / 1000
        position = pygame.Vector3(self.get_position())
        direction = pygame.Vector3(self.get_direction())

        if inputs.get_instance().mouse_was_pressed(button=1):
            self._mouse_down_xy = inputs.get_instance().mouse_pos()
            self._mouse_down_direction = direction

        if inputs.get_instance().mouse_is_dragging(button=1):
            screen_xy = renderengine.get_instance().get_game_size()
            start_xy, end_xy = inputs.get_instance().mouse_drag_total()
            dx = end_xy[0] - start_xy[0]
            dy = end_xy[1] - start_xy[1]

            dy_angle = dy / screen_xy[1] * self.get_fov()
            dx_angle = dx / screen_xy[0] * (self.get_fov() * screen_xy[0] / screen_xy[1])

            start_dir = self._mouse_down_direction
            dir_spherical = pygame.Vector3(start_dir.x, start_dir.z, start_dir.y).as_spherical()
            new_y_angle = dir_spherical[1] - dy_angle
            new_y_angle = util.bound(new_y_angle, *self.v_angle_limit)
            new_xz_angle = dir_spherical[2] - dx_angle

            temp = pygame.Vector3()
            temp.from_spherical((dir_spherical[0], new_y_angle, new_xz_angle))
            direction.x = temp[0]
            direction.y = temp[2]
            direction.z = temp[1]
        else:
            view_dx = inputs.get_instance().is_held_two_way(configs.ROTATE_LEFT, configs.ROTATE_RIGHT)
            if view_dx != 0:
                xz = pygame.Vector2(direction.x, direction.z)
                xz = xz.rotate(self.hrot_speed * dt_secs * view_dx)
                direction.x = xz[0]
                direction.z = xz[1]

            view_dy = inputs.get_instance().is_held_two_way(configs.ROTATE_DOWN, configs.ROTATE_UP)
            if view_dy != 0:
                dir_spherical = pygame.Vector3(direction.x, direction.z, direction.y).as_spherical()
                new_angle = dir_spherical[1] - self.vrot_speed * dt_secs * view_dy
                new_angle = util.bound(new_angle, *self.v_angle_limit)

                temp = pygame.Vector3()
                temp.from_spherical((dir_spherical[0], new_angle, dir_spherical[2]))
                direction.x = temp[0]
                direction.y = temp[2]
                direction.z = temp[1]

        ms = self.move_speed * dt_secs

        view_xz = pygame.Vector2(direction.x, direction.z)
        view_xz.scale_to_length(1)

        xz = pygame.Vector2(position.x, position.z)
        move_dxz = inputs.get_instance().is_held_four_way(left=configs.MOVE_LEFT, right=configs.MOVE_RIGHT,
                                                          posy=configs.MOVE_UP, negy=configs.MOVE_DOWN)
        if move_dxz[0] != 0:
            xz = xz + ms * view_xz.rotate(90 * move_dxz[0])
        if move_dxz[1] != 0:
            xz = xz + ms * view_xz.rotate(180 * (move_dxz[1] - 1) / 2)

        y = position.y
        move_dz = inputs.get_instance().is_held_two_way(configs.CROUCH, configs.JUMP)
        if move_dz != 0:
            y += self.fly_speed * dt_secs * move_dz

        self.set_direction(direction)
        self.set_position((xz[0], y, xz[1]))

        dfov = inputs.get_instance().was_pressed_two_way(configs.DEBUG_DECREASE_CAMERA_FOV,
                                                         configs.DEBUG_INCREASE_CAMERA_FOV)
        if dfov != 0:
            dfov = -5 if inputs.get_instance().shift_is_held() else 5
            self.set_fov(self.get_fov() + dfov)


class ThreeDeeLayer(layers.ImageLayer):

    def __init__(self, layer_id, layer_z):
        super().__init__(layer_id, layer_z)
        self.camera = Camera3D()
        self.use_perspective = True  # False for ortho
        self.show_textures = True
        self.show_wireframe = True

    def set_camera(self, cam):
        self.camera = cam.get_snapshot()

    def set_use_perspective(self, val):
        """Set whether to use a perspective or orthographic camera."""
        self.use_perspective = val

    def set_show_wireframe(self, val):
        self.show_wireframe = val

    def set_show_textures(self, val):
        self.show_textures = val

    def accepts_sprite_type(self, sprite_type):
        return sprite_type == sprites.SpriteTypes.THREE_DEE

    def populate_data_arrays(self, opaque_ids, translucent_ids, sprite_info_lookup, first_dirty_opaque_idx=0):
        pass  # we don't actually use these

    def get_sprites_grouped_by_model_id(self, engine):
        res = {}  # model_id -> list of Sprite3D
        for sprite_id in self.opaque_images:
            spr_3d = engine.sprite_info_lookup[sprite_id].sprite
            if spr_3d.model().get_model_id() not in res:
                res[spr_3d.model().get_model_id()] = []
            res[spr_3d.model().get_model_id()].append(spr_3d)
        return res

    def render(self, engine):
        if self.show_wireframe:
            self.render_internal(engine, wireframe=True)
        if self.show_textures:
            self.render_internal(engine)

    def render_internal(self, engine, wireframe=False):
        if not engine.is_opengl():
            return  # doesn't work in non-OpenGL mode, for obvious reasons

        self.set_client_states(True, engine, wireframe=wireframe)
        self._set_uniforms_for_scene(engine)

        model_ids_to_sprites = self.get_sprites_grouped_by_model_id(engine)
        for model_id in model_ids_to_sprites:
            # only pass model data (vertices, tex_coords, indices) once per unique model in the scene
            model = model_ids_to_sprites[model_id][0].model()
            self._pass_attributes_for_model(engine, model)

            # draw each sprite with that model, using the same data, but different uniforms
            for spr_3d in model_ids_to_sprites[model_id]:
                self._set_uniforms_for_sprite(engine, spr_3d, wireframe=wireframe)
                indices = self.opaque_data_arrays.indices
                glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)

        self.set_client_states(False, engine, wireframe=wireframe)

    def set_client_states(self, enable, engine, wireframe=False):
        engine.set_vertices_enabled(enable)
        engine.set_texture_coords_enabled(enable)
        engine.set_alpha_test_enabled(enable)
        engine.set_depth_test_enabled(enable)

        if enable:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)
            if wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            engine.set_global_color(None)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def get_view_matrix(self):
        target_pt = util.add(self.camera.get_position(), self.camera.get_direction())
        return matutils.get_matrix_looking_at2(self.camera.get_position(), target_pt, self.get_camera_up())

    def get_proj_matrix(self, engine):
        w, h = engine.get_game_size()
        znear, zfar = 0.5, 100000
        if self.use_perspective:
            return matutils.perspective_matrix(self.camera.get_fov() / 360 * 6.283, w / h, znear, zfar)
        else:
            zoom = 5 * self.camera.get_fov() / 45
            return matutils.ortho_matrix(-zoom, zoom, -zoom * h / w, zoom * h / w, znear, zfar)

    def get_camera_up(self):
        return self.camera.get_up_vector()

    def _set_uniforms_for_scene(self, engine):
        view = self.get_view_matrix()
        engine.set_view_matrix(view)

        proj = self.get_proj_matrix(engine)
        engine.set_proj_matrix(proj)

    def _set_uniforms_for_sprite(self, engine, spr_3d: 'Sprite3D', wireframe=False):
        model = spr_3d.get_xform(camera_pos=self.camera.get_position())
        engine.set_model_matrix(model)
        engine.set_global_color(spr_3d.color() if not wireframe else (0, 0, 0))

    def _pass_attributes_for_model(self, engine, model_3d):
        oda = self.opaque_data_arrays
        oda.vertices.resize(3 * len(model_3d.get_vertices()), refcheck=False)
        oda.tex_coords.resize(2 * len(model_3d.get_texture_coords()), refcheck=False)
        oda.indices.resize(len(model_3d.get_indices()), refcheck=False)

        model_3d.add_urself(oda.vertices, oda.tex_coords, oda.indices)

        engine.set_vertices(oda.vertices)
        engine.set_texture_coords(oda.tex_coords)


class Sprite3D(sprites.AbstractSprite):

    def __init__(self, model, layer_id, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), color=(1, 1, 1), uid=None):
        sprites.AbstractSprite.__init__(self, sprites.SpriteTypes.THREE_DEE, layer_id, uid=uid)
        self._model = model

        self._position = position   # location of the model's origin
        self._rotation = rotation   # rotation of the model in each plane w.r.t. the origin
        self._scale = scale         # scale of the model in each axis

        self._color = color

    def model(self) -> 'ThreeDeeModel':
        return self._model

    def get_xform(self, camera_pos=(0, 0, 0)):
        pos = self.position()
        scale = self.scale()

        # translation matrix
        T = numpy.identity(4, dtype=numpy.float32)
        T.itemset((3, 0), pos[0])
        T.itemset((3, 1), pos[1])
        T.itemset((3, 2), pos[2])
        T = T.transpose()  # this is weird T_T

        R = self.get_rotation_matrix(camera_pos=camera_pos)

        # scale matrix
        S = numpy.identity(4, dtype=numpy.float32)
        S.itemset((0, 0), scale[0])
        S.itemset((1, 1), scale[1])
        S.itemset((2, 2), scale[2])

        return T.dot(R).dot(S)

    def position(self):
        return self._position

    def x(self):
        return self._position[0]

    def y(self):
        return self._position[1]

    def z(self):
        return self._position[2]

    def rotation(self):
        return self._rotation

    def get_rotation_matrix(self, camera_pos=(0, 0, 0)):
        rot = self.get_effective_rotation(camera_pos=camera_pos)
        return matutils.rotation_matrix(rot, axis_order=(0, 1, 2))

    def get_effective_rotation(self, camera_pos=(0, 0, 0)):
        return self.rotation()

    def scale(self):
        return self._scale

    def color(self):
        return self._color

    def update(self, new_model=None,
               new_x=None, new_y=None, new_z=None, new_position=None,
               new_xrot=None, new_yrot=None, new_zrot=None, new_rotation=None,
               new_xscale=None, new_yscale=None, new_zscale=None, new_scale=None,
               new_color=None):
        did_change = False

        model = self._model
        if new_model is not None and new_model != self._model:
            did_change = True
            model = new_model

        position = [v for v in self._position]
        if new_position is not None:
            new_x = new_position[0]
            new_y = new_position[1]
            new_z = new_position[2]
        if new_x is not None and new_x != self._position[0]:
            position[0] = new_x
            did_change = True
        if new_y is not None and new_y != self._position[1]:
            position[1] = new_y
            did_change = True
        if new_z is not None and new_z != self._position[2]:
            position[2] = new_z
            did_change = True

        rotation = [v for v in self._rotation]
        if new_rotation is not None:
            new_xrot = new_rotation[0]
            new_yrot = new_rotation[1]
            new_zrot = new_rotation[2]
        if new_xrot is not None and new_xrot != self._rotation[0]:
            rotation[0] = new_xrot
            did_change = True
        if new_yrot is not None and new_yrot != self._rotation[1]:
            rotation[1] = new_yrot
            did_change = True
        if new_zrot is not None and new_zrot != self._rotation[2]:
            rotation[2] = new_zrot
            did_change = True

        scale = [v for v in self._scale]
        if new_scale is not None:
            if isinstance(new_scale, (int, float)):
                new_scale = (new_scale, new_scale, new_scale)
            new_xscale = new_scale[0]
            new_yscale = new_scale[1]
            new_zscale = new_scale[2]
        if new_xscale is not None and new_xscale != self._scale[0]:
            scale[0] = new_xscale
            did_change = True
        if new_yscale is not None and new_yscale != self._scale[1]:
            scale[1] = new_yscale
            did_change = True
        if new_zscale is not None and new_zscale != self._scale[2]:
            scale[2] = new_zscale
            did_change = True

        color = self._color
        if new_color is not None and new_color != self._color:
            color = new_color
            did_change = True

        if not did_change:
            return self
        else:
            return Sprite3D(model, self.layer_id(), position=position, rotation=rotation,
                            scale=scale, color=color, uid=self.uid())


class BillboardSprite3D(Sprite3D):

    def __init__(self, model, layer_id, horz_billboard=True, vert_billboard=False,
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), color=(1, 1, 1), uid=None):
        super().__init__(model, layer_id, position=position, rotation=rotation, scale=scale, color=color, uid=uid)
        self._horz_billboard = horz_billboard
        self._vert_billboard = vert_billboard

    def get_effective_rotation(self, camera_pos=(0, 0, 0)):
        towards_camera = util.set_length(util.sub(camera_pos, self.position()), 1)
        rot_to_camera = matutils.get_xyz_rot_to_face_direction(towards_camera,
                                                               do_pitch=self._vert_billboard,
                                                               do_yaw=self._horz_billboard)
        return util.add(self.rotation(), rot_to_camera)

    def update(self, new_model=None,
               new_x=None, new_y=None, new_z=None, new_position=None,
               new_xrot=None, new_yrot=None, new_zrot=None, new_rotation=None,
               new_xscale=None, new_yscale=None, new_zscale=None, new_scale=None,
               new_color=None, new_horz_billboard=None, new_vert_billboard=None):  # XXX just ignore this i know it's bad
        res = super().update(new_model, new_x=new_x, new_y=new_y, new_z=new_z, new_position=new_position,
                             new_xrot=new_xrot, new_yrot=new_yrot, new_zrot=new_zrot, new_rotation=new_rotation,
                             new_xscale=new_xscale, new_yscale=new_yscale, new_zscale=new_zscale, new_scale=new_scale,
                             new_color=new_color)

        did_change = False
        horz_billboard = self._horz_billboard
        if new_horz_billboard is not None and new_horz_billboard != self._horz_billboard:
            horz_billboard = new_horz_billboard
            did_change = True
        vert_billboard = self._vert_billboard
        if new_vert_billboard is not None and new_vert_billboard != self._vert_billboard:
            vert_billboard = new_vert_billboard
            did_change = True

        did_change |= (res.model() != self.model() or
                       res.position() != self.position() or
                       res.rotation() != self.rotation() or
                       res.scale() != self.scale() or
                       res.color() != self.color())

        if did_change:
            return BillboardSprite3D(res.model(), self.layer_id(),
                                     horz_billboard=horz_billboard, vert_billboard=vert_billboard,
                                     position=res.position(), rotation=res.rotation(), scale=res.scale(),
                                     color=res.color(), uid=self.uid())
        else:
            return self


class ThreeDeeModel:

    def __init__(self, model_id, vertices, normals, native_texture_coords, indices, map_from_texture_to_atlas=lambda xy: xy):
        """
        :param model_id: str
        :param vertices: list of (x, y, z)
        :param normals: list of (x, y, z)
        :param native_texture_coords: list of (x, y)
        :param indices: list of ints, one for each corner of each triangle
        :param map_from_texture_to_atlas: converts points from native_texture_coords to actual atlas coordinates
        """
        self._model_id = model_id

        self._vertices = vertices
        self._normals = normals
        self._native_texture_coords = native_texture_coords
        self._indices = indices

        self._map_from_texture_to_atlas = map_from_texture_to_atlas
        self._cached_atlas_coords = []  # list of (x, y)

    def get_model_id(self):
        return self._model_id

    def get_vertices(self):
        return self._vertices

    def get_indices(self):
        return self._indices

    def get_normals(self):
        return self._normals

    def get_texture_coords(self):
        if len(self._cached_atlas_coords) == 0:
            self._cached_atlas_coords = [self._map_from_texture_to_atlas(xy) for xy in self._native_texture_coords]
        return self._cached_atlas_coords

    def add_urself(self, vertices, tex_coords, indices):
        for i in range(0, 3 * len(self.get_vertices())):
            vertices[i] = self.get_vertices()[i // 3][i % 3]
        for i in range(0, 2 * len(self.get_texture_coords())):
            tex_coords[i] = self.get_texture_coords()[i // 2][i % 2]
        for i in range(0, len(self.get_indices())):
            indices[i] = self.get_indices()[i]

    @staticmethod
    def load_from_disk(model_id, model_path, map_from_texture_to_atlas):
        try:
            raw_vertices = []
            raw_normals = []
            raw_native_texture_coords = []
            triangle_faces = []

            safe_path = util.resource_path(model_path)
            with open(safe_path) as f:
                for line in f:
                    line = line.rstrip()  # remove trailing newlines and whitespace
                    if line.startswith("v "):
                        xyz = line[2:].split(" ")
                        vertex = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                        raw_vertices.append(vertex)

                    elif line.startswith("vn "):
                        xyz = line[3:].split(" ")
                        normal_vec = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                        raw_normals.append(normal_vec)

                    elif line.startswith("vt "):
                        xy = line[3:].split(" ")
                        texture_coords = (float(xy[0]), float(xy[1]))
                        raw_native_texture_coords.append(texture_coords)

                    elif line.startswith("f "):
                        corners = []
                        for corner in line[2:].split(" "):
                            vtn = corner.split("/")  # vertex, texture, normal
                            vertex_idx = int(vtn[0]) - 1
                            texture_idx = int(vtn[1]) - 1 if len(vtn) > 1 and len(vtn[1]) > 0 else -1
                            normal_idx = int(vtn[2]) - 1 if len(vtn) > 2 and len(vtn[2]) > 0 else -1
                            corners.append((vertex_idx, texture_idx, normal_idx))
                        triangle_faces.append(tuple(corners))

            vertices = []
            native_texture_coords = []
            normals = []
            indices = []

            for tri in triangle_faces:
                for c in tri:  # TODO use the normals
                    v_idx, t_idx, norm_idx = c
                    vertex_xyz = raw_vertices[v_idx]
                    texture_xy = raw_native_texture_coords[t_idx] if t_idx >= 0 else None
                    norm_xyz = raw_normals[norm_idx] if norm_idx >= 0 else None
                    # TODO can probably condense this a bit (only have one index per unique (vertex, texture, normal))
                    # TODO would that ever matter for most models? probably not?
                    index = len(indices)

                    vertices.append(vertex_xyz)
                    native_texture_coords.append(texture_xy)
                    normals.append(norm_xyz)
                    indices.append(index)
            print("INFO: loaded model ({} faces): {}".format(len(triangle_faces), model_path))
            return ThreeDeeModel(model_id, vertices, normals, native_texture_coords, indices,
                                 map_from_texture_to_atlas=map_from_texture_to_atlas)
        except IOError:
            print("ERROR: failed to load model: {}".format(model_path))
            return None

    @staticmethod
    def build_from_2d_model(model_2d: sprites.ImageModel) -> 'ThreeDeeModel':
        vertices = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0)]
        native_texture_coords = [(model_2d.tx1 if v[0] < 0 else model_2d.tx2,
                                  model_2d.ty1 if v[1] < 0 else model_2d.ty2) for v in vertices]
        normals = [(0, 0, 1)] * 6
        indices = [0, 1, 2, 3, 4, 5]

        return ThreeDeeModel("2d_sprite_" + str(model_2d.uid()), vertices, normals, native_texture_coords, indices)

