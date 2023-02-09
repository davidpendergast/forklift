import os
import typing

from OpenGL.GL import *

import numpy
import pygame
import math

import src.engine.layers as layers
import src.engine.sprites as sprites
import src.engine.spritesheets as spritesheets
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
        self.show_wireframe = False
        self.rainbow_mode = configs.rainbow_3d

        self.use_lighting = True
        self.ambient_lighting = (0.1, None)
        self.light_sources = []  # list of (xyz, color, diffuse_strength, specular_strength) for each light source

    def set_camera(self, cam):
        self.camera = cam.get_snapshot()

    def set_use_perspective(self, val):
        """Set whether to use a perspective or orthographic camera."""
        self.use_perspective = val

    def set_show_wireframe(self, val):
        self.show_wireframe = val

    def set_show_textures(self, val):
        self.show_textures = val

    def set_rainbow_mode(self, val):
        self.rainbow_mode = val

    def set_use_lighting(self, val):
        self.use_lighting = val

    def set_ambient_lighting(self, strength, color):
        self.ambient_lighting = (strength, color)

    def set_light_sources(self, sources):
        """Sets the light sources for the scene.
            sources: list of (xyz, color=(1., 1., 1.), diffuse_strength=1.0, specular_strength=0.5)
        """
        self.light_sources.clear()
        for src in sources:
            xyz = src[0]
            color = src[1] if len(src) > 1 else (1., 1., 1.)
            diffuse_strength = src[2] if len(src) > 2 else 1.0
            specular_strength = src[3] if len(src) > 3 else 0.5
            self.light_sources.append((xyz, color, diffuse_strength, specular_strength))

    def accepts_sprite_type(self, sprite_type):
        return sprite_type == sprites.SpriteTypes.THREE_DEE

    def populate_data_arrays(self, opaque_ids, translucent_ids, sprite_info_lookup, first_dirty_opaque_idx=0):
        pass  # we don't actually use these

    def get_sprites_grouped_by_model_id(self, engine) -> typing.Dict[str, typing.List['Sprite3D']]:
        res = {}  # model_id -> list of Sprite3D
        for sprite_id in self.opaque_images:
            spr_3d = engine.sprite_info_lookup[sprite_id].sprite
            model_id = spr_3d.model().get_model_id()
            if model_id not in res:
                res[model_id] = []
            res[model_id].append(spr_3d)
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

        # collect all meshes in the scene and their parent sprite
        meshes_to_render: typing.Dict[typing.Tuple['ThreeDeeMesh', sprites.ImageModel], typing.Set[Sprite3D]] = {}

        for sprite_id in self.opaque_images:
            spr_3d = engine.sprite_info_lookup[sprite_id].sprite

            texture = spr_3d.texture()
            if texture is None:
                texture = spritesheets.get_white_square_img()
            if self.rainbow_mode:
                texture = spritesheets.get_rainbow_img()

            for mesh in spr_3d.model().all_unique_meshes():
                key = (mesh, texture)
                if key not in meshes_to_render:
                    meshes_to_render[key] = set()
                meshes_to_render[key].add(spr_3d)

        for (mesh, texture) in meshes_to_render:
            # only pass raw mesh data (vertices, normals, tex_coords, indices)
            # once per unique mesh in the scene
            self._pass_attributes_for_mesh(engine, mesh, texture)

            for spr_3d in meshes_to_render[(mesh, texture)]:
                for mesh_name in spr_3d.model().all_names_for_mesh(mesh):
                    spr_3d.set_uniforms_for_mesh(mesh_name, engine, self.camera.get_position(), wireframe=wireframe)
                    indices = self.opaque_data_arrays.indices
                    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)

        self.set_client_states(False, engine, wireframe=wireframe)

    def set_client_states(self, enable, engine, wireframe=False):
        engine.set_vertices_enabled(enable)
        engine.set_normals_enabled(enable)
        engine.set_texture_coords_enabled(enable)
        engine.set_alpha_test_enabled(enable)
        engine.set_depth_test_enabled(enable)

        if enable:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)
            if wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                engine.set_use_lighting(self.use_lighting)
                engine.set_ambient_lighting(*self.ambient_lighting)

                if len(self.light_sources) > 0:
                    # TODO multiple light sources
                    xyz, color, diff, spec = self.light_sources[0]
                    engine.set_diffuse_lighting(xyz, strength=diff, specular_strength=spec, color=color)
                else:
                    engine.set_diffuse_lighting(None)
        else:
            engine.set_global_color(None)
            engine.set_use_lighting(False)
            engine.set_ambient_lighting(None)
            engine.set_diffuse_lighting(None)
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
        eye = self.camera.get_position()
        engine.set_eye_pos(eye)

        view = self.get_view_matrix()
        engine.set_view_matrix(view)

        proj = self.get_proj_matrix(engine)
        engine.set_proj_matrix(proj)

    def _set_uniforms_for_sprite(self, engine, spr_3d: 'Sprite3D', wireframe=False):
        model = spr_3d.get_xform(camera_pos=self.camera.get_position())
        engine.set_model_matrix(model)
        engine.set_global_color(spr_3d.color() if not wireframe else (0, 0, 0))

    def _pass_attributes_for_mesh(self, engine, mesh_3d, texture):
        oda = self.opaque_data_arrays
        oda.vertices.resize(3 * len(mesh_3d.get_vertices()), refcheck=False)
        oda.normals.resize(3 * len(mesh_3d.get_normals()), refcheck=False)
        oda.tex_coords.resize(2 * len(mesh_3d.get_texture_coords(texture)), refcheck=False)
        oda.indices.resize(len(mesh_3d.get_indices()), refcheck=False)

        mesh_3d.add_urself(texture, oda.vertices, oda.normals, oda.tex_coords, oda.indices)

        engine.set_vertices(oda.vertices)
        engine.set_normals(oda.normals)
        engine.set_texture_coords(oda.tex_coords)


class Sprite3D(sprites.AbstractSprite):

    def __init__(self, layer_id, model: 'ThreeDeeModel' = None, texture: sprites.ImageModel = None,
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1),
                 color=(1, 1, 1), mesh_xforms=None, uid=None):
        sprites.AbstractSprite.__init__(self, sprites.SpriteTypes.THREE_DEE, layer_id, uid=uid)
        self._model = model
        self._texture = texture

        self._position = position   # location of the model's origin
        self._rotation = rotation   # rotation of the model in each plane w.r.t. the origin
        self._scale = scale         # scale of the model in each axis

        self._color = color

        self._extra_mesh_xforms = {} if mesh_xforms is None else mesh_xforms  # mesh_name -> (pos, scale, rot)

    def model(self) -> 'ThreeDeeModel':
        return self._model

    def texture(self) -> sprites.ImageModel:
        return self._texture

    @staticmethod
    def _calc_xform(pos=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0)):
        # translation matrix
        T = numpy.identity(4, dtype=numpy.float32)
        T.itemset((3, 0), pos[0])
        T.itemset((3, 1), pos[1])
        T.itemset((3, 2), pos[2])
        T = T.transpose()  # this is weird T_T

        R = matutils.rotation_matrix(rot, axis_order=(0, 1, 2))

        # scale matrix
        S = numpy.identity(4, dtype=numpy.float32)
        S.itemset((0, 0), scale[0])
        S.itemset((1, 1), scale[1])
        S.itemset((2, 2), scale[2])

        return T.dot(R).dot(S)

    def get_xform(self, camera_pos=(0, 0, 0)):
        pos = self.position()
        scale = self.scale()
        rot = self.get_effective_rotation(camera_pos=camera_pos)
        return Sprite3D._calc_xform(pos, scale, rot)

    def update_mesh(self, name='base', new_pos=None, new_scale=None, new_rot=None) -> 'Sprite3D':
        pos = (0, 0, 0)
        scale = (1, 1, 1)
        rot = (0, 0, 0)
        if name in self._extra_mesh_xforms:
            pos, scale, rot = self._extra_mesh_xforms[name]

        did_change = False
        if new_pos is not None and new_pos != pos:
            did_change = True
            pos = new_pos
        if new_scale is not None and new_scale != scale:
            did_change = True
            scale = new_scale
        if new_rot is not None and new_rot != rot:
            did_change = True
            rot = new_rot

        if not did_change:
            return self
        else:
            copy = self.update(force_new=True)
            mesh_xforms_copy = dict(self._extra_mesh_xforms)
            mesh_xforms_copy[name] = (pos, scale, rot)
            copy._extra_mesh_xforms = mesh_xforms_copy
            return copy

    def get_mesh_xform(self, name='base'):
        if name in self._extra_mesh_xforms:
            pos, scale, rot = self._extra_mesh_xforms[name]
            return self._calc_xform(pos, scale, rot)
        return numpy.identity(4, dtype=numpy.float32)

    def get_extra_mesh_xforms(self):
        return self._extra_mesh_xforms

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

    def get_effective_rotation(self, camera_pos=(0, 0, 0)):
        return self.rotation()

    def scale(self):
        return self._scale

    def color(self):
        return self._color

    def set_uniforms_for_mesh(self, mesh_name, engine, camera_pos, wireframe=False):
        model = self.model()
        base_xform = model.get_xform(mesh_name) if model is not None else IDENTITY
        extra_xform = self.get_mesh_xform(name=mesh_name)
        sprite_xform = self.get_xform(camera_pos=camera_pos)

        engine.set_model_matrix(sprite_xform @ extra_xform @ base_xform)
        engine.set_global_color(self.color() if not wireframe else (0, 0, 0))

    def update(self, new_model=None, new_texture=None,
               new_x=None, new_y=None, new_z=None, new_position=None,
               new_xrot=None, new_yrot=None, new_zrot=None, new_rotation=None,
               new_xscale=None, new_yscale=None, new_zscale=None, new_scale=None,
               new_color=None, force_new=False):
        did_change = False

        model = self._model
        if new_model is not None and new_model != self._model:
            did_change = True
            model = new_model

        texture = self._texture
        if new_texture is not None and new_texture != self._texture:
            did_change = True
            texture = new_texture

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

        if not did_change and not force_new:
            return self
        else:
            return Sprite3D(self.layer_id(), model=model, texture=texture,
                            position=position, rotation=rotation, scale=scale,
                            color=color, uid=self.uid())

    def interpolate(self, other: 'Sprite3D', t, force_new=False) -> 'Sprite3D':
        func = util.linear_interp
        res = self.update(new_position=func(self.position(), other.position(), t),
                          new_rotation=func(self.rotation(), other.rotation(), t, loop=2*math.pi),
                          new_scale=func(self.scale(), other.scale(), t),
                          new_color=func(self.color(), other.color(), util.bound(t, 0., 1.)),
                          force_new=force_new)

        interp_mesh_xforms = {}
        my_mesh_xforms = self.get_extra_mesh_xforms()
        ot_mesh_xforms = other.get_extra_mesh_xforms()
        for name in (my_mesh_xforms.keys() | ot_mesh_xforms.keys()):
            my_trans, my_rot, my_scale = my_mesh_xforms[name] if name in my_mesh_xforms else ((0, 0, 0), (1, 1, 1), (0, 0, 0))
            ot_trans, ot_rot, ot_scale = ot_mesh_xforms[name] if name in ot_mesh_xforms else ((0, 0, 0), (1, 1, 1), (0, 0, 0))
            interp_mesh_xforms[name] = (func(my_trans, ot_trans, t),
                                        func(my_rot, ot_rot, t, loop=2*math.pi),
                                        func(my_scale, ot_scale, t))
        res._extra_mesh_xforms = interp_mesh_xforms

        return res


class BillboardSprite3D(Sprite3D):

    def __init__(self, layer_id, model=None, horz_billboard=True, vert_billboard=False,
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), color=(1, 1, 1), uid=None):
        super().__init__(layer_id, model=model, position=position, rotation=rotation, scale=scale, color=color, uid=uid)
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
            return BillboardSprite3D(self.layer_id(), res.model(),
                                     horz_billboard=horz_billboard, vert_billboard=vert_billboard,
                                     position=res.position(), rotation=res.rotation(), scale=res.scale(),
                                     color=res.color(), uid=self.uid())
        else:
            return self


IDENTITY = numpy.identity(4)


class ThreeDeeModel:

    def __init__(self, model_id):
        self._model_id = model_id

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self.get_model_id() == other.get_model_id()

    def __hash__(self):
        return hash(self.get_model_id())

    def get_model_id(self):
        return self._model_id

    def all_mesh_names(self) -> typing.Generator[str, None, None]:
        raise NotImplementedError()

    def all_names_for_mesh(self, mesh) -> typing.Set['str']:
        raise NotImplementedError()

    def apply_translation(self, xyz_trans, name=None):
        mat = matutils.translation_matrix(xyz_trans)
        return self.apply_xform(mat, name=name)

    def apply_rotation(self, xyz_rot, name=None):
        mat = matutils.rotation_matrix(xyz_rot)
        return self.apply_xform(mat, name=name)

    def apply_scaling(self, xyz_scale, name=None):
        mat = matutils.scale_matrix(xyz_scale)
        return self.apply_xform(mat, name=name)

    def apply_xform(self, xform, name=None) -> 'ThreeDeeModel':
        raise NotImplementedError()

    def get_xform(self, name='base') -> numpy.ndarray:
        raise NotImplementedError()

    def get_mesh(self, name='base') -> 'ThreeDeeMesh':
        raise NotImplementedError()

    def all_unique_meshes(self) -> typing.Generator['ThreeDeeMesh', None, None]:
        raise NotImplementedError()


class ThreeDeeMultiMesh(ThreeDeeModel):

    def __init__(self, model_id, meshes: typing.Dict[str, typing.Union['ThreeDeeMesh',
                                                                       typing.Tuple['ThreeDeeMesh', numpy.ndarray]]]):
        super().__init__(model_id)
        self._name_to_mesh_and_xform_lookup: typing.Dict[str, typing.Tuple['ThreeDeeMesh', numpy.ndarray]] = {}
        self._mesh_to_names_lookup: typing.Dict['ThreeDeeMesh', typing.Set[str]] = {}

        # just a little type-checking
        for mesh_name in meshes:
            if not isinstance(mesh_name, str):
                raise TypeError(f"invalid mesh name: {mesh_name} ({type(mesh_name).__name__})")

            val = meshes[mesh_name]
            if isinstance(val, ThreeDeeMesh):
                val = (val, IDENTITY)

            if not isinstance(val[0], ThreeDeeMesh) or not isinstance(val[1], numpy.ndarray):
                raise TypeError(f"invalid mesh value: ({val[0]} ({type(val[0]).__name__}), "
                                f"{val[1]} ({type(val[1]).__name__})")
            else:
                self._name_to_mesh_and_xform_lookup[mesh_name] = val

            if val[0] not in self._mesh_to_names_lookup:
                self._mesh_to_names_lookup[val[0]] = set()
            self._mesh_to_names_lookup[val[0]].add(mesh_name)

    def apply_xform(self, xform, name=None) -> 'ThreeDeeMultiMesh':
        names_to_xform = [name] if name is not None else list(self.all_mesh_names())
        for n in names_to_xform:
            mesh, old_xform = self._name_to_mesh_and_xform_lookup[n]
            self._name_to_mesh_and_xform_lookup[n] = (mesh, xform @ old_xform)
        return self

    def all_mesh_names(self) -> typing.Generator[str, None, None]:
        for name in self._name_to_mesh_and_xform_lookup:
            yield name

    def all_names_for_mesh(self, mesh) -> typing.Set['str']:
        if mesh in self._mesh_to_names_lookup:
            return self._mesh_to_names_lookup[mesh]
        return set()

    def get_mesh(self, name="base") -> 'ThreeDeeMesh':
        return self._name_to_mesh_and_xform_lookup[name][0]

    def get_xform(self, name="base") -> numpy.ndarray:
        return self._name_to_mesh_and_xform_lookup[name][1]

    def all_unique_meshes(self) -> typing.Generator['ThreeDeeMesh', None, None]:
        for mesh in self._mesh_to_names_lookup:
            yield mesh

    @staticmethod
    def load_from_disk(model_id, path_to_dir, mesh_name_mapping) -> 'ThreeDeeMultiMesh':
        meshes = {}
        for mesh_name in mesh_name_mapping:
            filename = mesh_name_mapping[mesh_name]
            mesh_id = filename[:-4] + f"_({model_id})"
            meshes[mesh_name] = ThreeDeeMesh.load_from_disk(mesh_id, os.path.join(path_to_dir, filename))
        return ThreeDeeMultiMesh(model_id, meshes)


class ThreeDeeMesh(ThreeDeeModel):

    def __init__(self, mesh_id, vertices, normals, native_texture_coords, indices):
        """
        :param model_id: str
        :param vertices: list of (x, y, z)
        :param normals: list of (x, y, z)
        :param native_texture_coords: list of (x, y)
        :param indices: list of ints, one for each corner of each triangle
        """
        super().__init__(mesh_id + "_(model)")
        self._mesh_id = mesh_id

        self._vertices = vertices
        self._normals = normals
        self._native_texture_coords = native_texture_coords
        self._indices = indices

        self._base_xform = IDENTITY

        self._cached_atlas_coords = {}  # ImageModel -> list of xy

    def get_mesh_id(self):
        return self._mesh_id

    def get_vertices(self):
        return self._vertices

    def get_indices(self):
        return self._indices

    def get_normals(self):
        return self._normals

    def get_texture_coords(self, texture: sprites.ImageModel):
        if texture not in self._cached_atlas_coords:
            self._cached_atlas_coords[texture] = [texture.to_atlas_coords(txy) for txy in self._native_texture_coords]
        return self._cached_atlas_coords[texture]

    def apply_xform(self, xform, name='base') -> 'ThreeDeeMesh':
        self._base_xform = xform @ self._base_xform
        return self

    def get_xform(self, name='base') -> numpy.ndarray:
        return self._base_xform

    def add_urself(self, texture: sprites.ImageModel, vertices, normals, tex_coords, indices):
        for i in range(0, 3 * len(self.get_vertices())):
            vertices[i] = self.get_vertices()[i // 3][i % 3]
        for i in range(0, 3 * len(self.get_normals())):
            normals[i] = self.get_normals()[i // 3][i % 3]
        for i in range(0, 2 * len(self.get_texture_coords(texture))):
            tex_coords[i] = self.get_texture_coords(texture)[i // 2][i % 2]
        for i in range(0, len(self.get_indices())):
            indices[i] = self.get_indices()[i]

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self.get_mesh_id() == other.get_mesh_id()

    def __hash__(self):
        return hash(self.get_mesh_id())

    def all_mesh_names(self) -> typing.Generator[str, None, None]:
        yield 'base'

    def all_names_for_mesh(self, mesh) -> typing.Set['str']:
        return {'base'}

    def get_mesh(self, name='base') -> 'ThreeDeeMesh':
        return self

    def all_unique_meshes(self) -> typing.Generator['ThreeDeeMesh', None, None]:
        yield self

    @staticmethod
    def load_from_disk(model_id, mesh_path):
        try:
            raw_vertices = []
            raw_normals = []
            raw_native_texture_coords = []
            triangle_faces = []

            with open(mesh_path) as f:
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
                for c in tri:
                    v_idx, t_idx, norm_idx = c
                    vertex_xyz = raw_vertices[v_idx]
                    texture_xy = raw_native_texture_coords[t_idx] if t_idx >= 0 else None
                    norm_xyz = raw_normals[norm_idx] if norm_idx >= 0 else None
                    index = len(indices)

                    vertices.append(vertex_xyz)
                    native_texture_coords.append(texture_xy)
                    normals.append(norm_xyz)
                    indices.append(index)
            print("INFO: loaded model ({} faces): {}".format(len(triangle_faces), mesh_path))
            return ThreeDeeMesh(model_id, vertices, normals, native_texture_coords, indices)
        except IOError:
            print("ERROR: failed to load model: {}".format(mesh_path))
            return None

    @staticmethod
    def build_from_2d_model(model_2d: sprites.ImageModel) -> 'ThreeDeeMesh':
        vertices = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0)]
        native_texture_coords = [(model_2d.tx1 if v[0] < 0 else model_2d.tx2,
                                  model_2d.ty1 if v[1] < 0 else model_2d.ty2) for v in vertices]
        normals = [(0, 0, 1)] * 6
        indices = [0, 1, 2, 3, 4, 5]

        return ThreeDeeMesh("2d_sprite_" + str(model_2d.uid()), vertices, normals, native_texture_coords, indices)



