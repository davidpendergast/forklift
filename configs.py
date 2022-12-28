
import os.path

name_of_game = "Forklift Game"
version = "0.0.0"
userdata_subdir = "forklift"
runtime_icon_path = os.path.join("assets", "icon.png")


""" Display """
default_window_size = (640, 480)  # size of window when the game starts.
minimum_window_size = (320, 240)  # if the window is smaller than this, it will begin cropping the picture.

allow_fullscreen = True
allow_window_resize = True

clear_color = (0.666, 0.666, 0.666)


""" Pixel Scaling """
optimal_window_size = (640, 480)
optimal_pixel_scale = 1  # how many screen pixels each "game pixel" will take up at the optimal window size.

auto_resize_pixel_scale = True  # whether to automatically update the pixel scale as the window grows and shrinks.
minimum_auto_pixel_scale = 1


""" FPS """
target_fps = 60
precise_fps = False


""" Miscellaneous """
start_in_compat_mode = False
do_crash_reporting = True  # whether to produce a crash file when the program exits via an exception.

is_dev = os.path.exists(".gitignore")  # yikes

key_repeat_delay = 30  # keys held for longer than this many ticks will start to be typed repeatedly
key_repeat_period = 5  # after the delay has passed, the key will be typed every X ticks until released


""" Keybinds """
MOVE_UP = "up"
MOVE_LEFT = "left"
MOVE_RIGHT = "right"
MOVE_DOWN = "down"
JUMP = "jump"
CROUCH = "crouch"

ENTER = "enter"
ESCAPE = "escape"

ROTATE_UP = "rot_up"
ROTATE_LEFT = "rot_left"
ROTATE_RIGHT = "rot_right"
ROTATE_DOWN = "rot_down"

DEBUG_TOGGLE_WIREFRAME = "toggle_wireframe"
DEBUG_TOGGLE_TEXTURES = "toggle_textures"
DEBUG_INCREASE_CAMERA_FOV = "increase_fov"
DEBUG_DECREASE_CAMERA_FOV = "decrease_fov"
DEBUG_TOGGLE_FREE_CAMERA = "toggle_freecam"
DEBUG_TOGGLE_ORTHO_CAMERA = "toggle_ortho_camera"


""" 3D Debug """
rainbow_3d = False

