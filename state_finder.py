import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath('/'))
from utils import load_toml_as_dict

orig_screen_width, orig_screen_height = 1920, 1080

states_path = r"./images/states/"

star_drops_path = r"./images/star_drop_types/"
images_with_star_drop = []
if os.path.isdir(star_drops_path):
    for file in os.listdir(star_drops_path):
        if "star_drop" in file:
            images_with_star_drop.append(file)

end_results_path = r"./images/end_results/"

region_data = load_toml_as_dict("./cfg/lobby_config.toml")['template_matching']
super_debug = load_toml_as_dict("./cfg/general_config.toml")['super_debug'] == "yes"
if super_debug:
    debug_folder = "./debug_frames/"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

# (template_path, frame_width, frame_height) -> resized template stored in RGB
# so that we can match directly against scrcpy's RGB frames without paying a
# full-frame BGR <-> RGB conversion every state check.
cached_templates = {}
# (frame_width, frame_height, region_tuple) -> (x, y, w, h) precomputed crop
# rectangle so we avoid a few multiplies + int casts every state check.
_cached_regions = {}


def load_template(image_path, width, height):
    key = (image_path, width, height)
    cached = cached_templates.get(key)
    if cached is not None:
        return cached
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Template not found: {image_path}")
    # Templates are stored as BGR by cv2.imread but the frames we match
    # against are RGB, so flip the template once at cache-time. matchTemplate
    # then operates in the same color space without any per-frame conversion.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    width_ratio = width / orig_screen_width
    height_ratio = height / orig_screen_height
    new_size = (max(1, int(orig_width * width_ratio)), max(1, int(orig_height * height_ratio)))
    resized_image = cv2.resize(image, new_size)
    if not resized_image.flags['C_CONTIGUOUS']:
        resized_image = np.ascontiguousarray(resized_image)
    cached_templates[key] = resized_image
    return resized_image


def _crop_for_region(image, region):
    height, width = image.shape[:2]
    # ``region`` arrives from TOML as a list (unhashable); convert once so we
    # can use it as a dict key.
    region_key = tuple(region)
    key = (width, height, region_key)
    rect = _cached_regions.get(key)
    if rect is None:
        orig_x, orig_y, orig_width, orig_height = region_key
        width_ratio = width / orig_screen_width
        height_ratio = height / orig_screen_height
        new_x = int(orig_x * width_ratio)
        new_y = int(orig_y * height_ratio)
        new_w = int(orig_width * width_ratio)
        new_h = int(orig_height * height_ratio)
        rect = (new_x, new_y, new_w, new_h)
        _cached_regions[key] = rect
    x, y, w, h = rect
    return image[y:y + h, x:x + w]


def is_template_in_region(image, template_path, region):
    height, width = image.shape[:2]
    cropped_image = _crop_for_region(image, region)
    loaded_template = load_template(template_path, width, height)
    if cropped_image.size == 0 or loaded_template.size == 0:
        return False
    # matchTemplate requires the search image to be at least as large as the
    # template in both dimensions; bail out cheaply otherwise instead of
    # raising a cv2 error.
    th, tw = loaded_template.shape[:2]
    if cropped_image.shape[0] < th or cropped_image.shape[1] < tw:
        return False
    result = cv2.matchTemplate(cropped_image, loaded_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > 0.7


crop_region = load_toml_as_dict("./cfg/lobby_config.toml")['lobby']['trophy_observer']


def find_game_result(screenshot):
    if is_template_in_region(screenshot, end_results_path + 'victory.png', crop_region):
        return "victory"
    if is_template_in_region(screenshot, end_results_path + 'defeat.png', crop_region):
        return "defeat"
    if is_template_in_region(screenshot, end_results_path + 'draw.png', crop_region):
        return "draw"
    return False


def get_in_game_state(image):
    game_result = is_in_end_of_a_match(image)
    if game_result:
        return f"end_{game_result}"
    if is_in_shop(image):
        return "shop"
    if is_in_offer_popup(image):
        return "popup"
    if is_in_lobby(image):
        return "lobby"
    if is_in_brawler_selection(image):
        return "brawler_selection"
    if is_in_brawl_pass(image) or is_in_star_road(image):
        return "shop"
    if is_in_star_drop(image):
        return "star_drop"
    if is_in_trophy_reward(image):
        return "trophy_reward"
    return "match"


def is_in_shop(image) -> bool:
    return is_template_in_region(image, states_path + 'powerpoint.png', region_data["powerpoint"])


def is_in_brawler_selection(image) -> bool:
    return is_template_in_region(image, states_path + 'brawler_menu_task.png', region_data["brawler_menu_task"])


def is_in_offer_popup(image) -> bool:
    return is_template_in_region(image, states_path + 'close_popup.png', region_data["close_popup"])


def is_in_lobby(image) -> bool:
    return is_template_in_region(image, states_path + 'lobby_menu.png', region_data["lobby_menu"])


def is_in_end_of_a_match(image):
    return find_game_result(image)


def is_in_trophy_reward(image):
    return is_template_in_region(image, states_path + 'trophies_screen.png', region_data["trophies_screen"])


def is_in_brawl_pass(image):
    return is_template_in_region(image, states_path + 'brawl_pass_house.PNG', region_data['brawl_pass_house'])


def is_in_star_road(image):
    return is_template_in_region(image, states_path + "go_back_arrow.png", region_data['go_back_arrow'])


def is_in_star_drop(image):
    for image_filename in images_with_star_drop:
        if is_template_in_region(image, star_drops_path + image_filename, region_data['star_drop']):
            return True
    return False


def get_state(screenshot):
    # Frames arrive from scrcpy already in RGB. Templates are cached in RGB on
    # load, so we no longer pay a 1920x1080 cv2.cvtColor on every state check.
    if super_debug:
        cv2.imwrite(
            f"./debug_frames/state_screenshot_{len(os.listdir('./debug_frames'))}.png",
            cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR),
        )
    state = get_in_game_state(screenshot)
    print(f"State: {state}")
    return state
