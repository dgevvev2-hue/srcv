import asyncio
import time

import window_controller
from gui.hub import Hub
from gui.login import login
from gui.main import App
from gui.select_brawler import SelectBrawler
from lobby_automation import LobbyAutomation
from play import Play
from stage_manager import StageManager
from state_finder import get_state
from time_management import TimeManagement
from utils import load_toml_as_dict, current_wall_model_is_latest, api_base_url
from utils import get_brawler_list, update_missing_brawlers_info, check_version, async_notify_user, \
    update_wall_model_classes, get_latest_wall_model_file, get_latest_version, cprint
from window_controller import WindowController

pyla_version = load_toml_as_dict("./cfg/general_config.toml")['pyla_version']


def pyla_main(data):
    class Main:

        def __init__(self):
            self.window_controller = WindowController()
            self.Play = Play(*self.load_models(), self.window_controller)
            self.Time_management = TimeManagement()
            self.lobby_automator = LobbyAutomation(self.window_controller)
            self.Stage_manager = StageManager(data, self.lobby_automator, self.window_controller)
            self.states_requiring_data = ["lobby"]
            if data[0]['automatically_pick']:
                print("Picking brawler automatically")
                self.lobby_automator.select_brawler(data[0]['brawler'])
            self.Play.current_brawler = data[0]['brawler']
            self.no_detections_action_threshold = 60 * 8
            self.initialize_stage_manager()
            self.state = None
            try:
                self.max_ips = int(load_toml_as_dict("cfg/general_config.toml")['max_ips'])
            except ValueError:
                self.max_ips = None
            self.run_for_minutes = int(load_toml_as_dict("cfg/general_config.toml")['run_for_minutes'])
            self.start_time = time.time()
            self.time_to_stop = False
            self.in_cooldown = False
            self.cooldown_start_time = 0
            self.cooldown_duration = 3 * 60

        def initialize_stage_manager(self):
            self.Stage_manager.Trophy_observer.win_streak = data[0]['win_streak']
            self.Stage_manager.Trophy_observer.current_trophies = data[0]['trophies']
            self.Stage_manager.Trophy_observer.current_wins = data[0]['wins'] if data[0]['wins'] != "" else 0

        @staticmethod
        def load_models():
            folder_path = "./models/"
            model_names = ['mainInGameModel.onnx', 'tileDetector.onnx']
            loaded_models = []

            for name in model_names:
                loaded_models.append(folder_path + name)
            return loaded_models

        def restart_brawl_stars(self):
            self.window_controller.restart_brawl_stars()
            self.Play.time_since_detections["player"] = time.time()
            self.Play.time_since_detections["enemy"] = time.time()
            if self.window_controller.device.app_current().package != window_controller.BRAWL_STARS_PACKAGE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    screenshot = self.window_controller.screenshot()
                    loop.run_until_complete(async_notify_user("bot_is_stuck", screenshot))
                finally:
                    loop.close()
                print("Bot got stuck. User notified. Shutting down.")
                self.window_controller.keys_up(list("wasd"))
                self.window_controller.close()
                import sys
                sys.exit(1)

        def manage_time_tasks(self, frame, current_time):
            if self.Time_management.state_check(current_time):
                state = get_state(frame)
                self.state = state
                if state != "match":
                    self.Play.time_since_last_proceeding = current_time
                self.Stage_manager.do_state(state, None)

            if self.Time_management.no_detections_check(current_time):
                threshold = self.no_detections_action_threshold
                for value in self.Play.time_since_detections.values():
                    if current_time - value > threshold:
                        self.restart_brawl_stars()
                        break

            if self.Time_management.idle_check(current_time):
                self.lobby_automator.check_for_idle(frame)

        def main(self):
            s_time = time.time()
            c = 0
            target_period = (1.0 / self.max_ips) if self.max_ips else 0.0
            stale_timeout = self.window_controller.FRAME_STALE_TIMEOUT
            run_for_seconds = self.run_for_minutes * 60 if self.run_for_minutes > 0 else 0

            while True:
                frame_start = time.perf_counter()
                current_time = time.time()

                if run_for_seconds and not self.in_cooldown:
                    if current_time - self.start_time >= run_for_seconds:
                        cprint(
                            f"timer is done, {self.run_for_minutes} is over. continuing for 3 minutes if in game",
                            "#AAE5A4",
                        )
                        self.in_cooldown = True  # tries to finish game if in game
                        self.cooldown_start_time = current_time
                        self.Stage_manager.states['lobby'] = lambda: 0

                if self.in_cooldown and current_time - self.cooldown_start_time >= self.cooldown_duration:
                    cprint("stopping bot fully", "#AAE5A4")
                    break

                # IPS reporting once per second using cached current_time.
                if current_time - s_time > 1:
                    elapsed = current_time - s_time
                    if elapsed > 0:
                        print(f"{c / elapsed:.2f} IPS")
                    s_time = current_time
                    c = 0

                frame = self.window_controller.screenshot()

                # Reuse the last frame timestamp recorded inside screenshot()
                # so we avoid acquiring the scrcpy frame lock a second time
                # just to check freshness.
                last_ft = self.window_controller.last_frame_time
                if last_ft > 0 and (current_time - last_ft) > stale_timeout:
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("Stale frame detected -- restarting the game.")
                    self.window_controller.restart_brawl_stars()

                self.manage_time_tasks(frame, current_time)

                brawler = self.Stage_manager.brawlers_pick_data[0]['brawler']
                self.Play.main(frame, brawler, self, current_time)
                if not getattr(self.Play, '_frame_skipped', False):
                    c += 1

                if target_period:
                    work_time = time.perf_counter() - frame_start
                    if work_time < target_period:
                        time.sleep(target_period - work_time)

    main = Main()
    main.main()


all_brawlers = get_brawler_list()
if api_base_url != "localhost":
    update_missing_brawlers_info(all_brawlers)
    check_version()
    update_wall_model_classes()
    if not current_wall_model_is_latest():
        print("New Wall detection model found, downloading... (this might take a few minutes depending on your internet speed)")
        get_latest_wall_model_file()

# --- Auto-update from GitHub ---
_auto_update_cfg = load_toml_as_dict("cfg/general_config.toml").get("auto_update", "no")
if _auto_update_cfg in ("yes", "check"):
    try:
        from tools.auto_updater import check_for_update, auto_update_on_startup
        if _auto_update_cfg == "check":
            _, _, _msg = check_for_update()
            print(_msg)
        else:
            auto_update_on_startup()
    except Exception as _exc:
        print(f"Auto-update skipped: {_exc}")

# Use the smaller ratio to maintain aspect ratio
app = App(login, SelectBrawler, pyla_main, all_brawlers, Hub)
app.start(pyla_version, get_latest_version)
