"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

from berkie_reachy.gstreamer_env import configure_gstreamer_bundle_env

configure_gstreamer_bundle_env()

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from berkie_reachy.config import config
from berkie_reachy.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from berkie_reachy.moves import MovementManager
    from berkie_reachy.console import LocalStream
    from berkie_reachy.openai_realtime import OpenaiRealtimeHandler
    from berkie_reachy.berky_live import BerkyLiveHandler
    from berkie_reachy.tools.core_tools import ToolDependencies
    from berkie_reachy.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    # Shared even when Welcomer is disabled (cheap, and keeps BerkyLiveHandler's
    # constructor signature uniform) - see interaction_mode.py. Only actually gates
    # anything once Welcomer is running alongside it.
    from berkie_reachy.interaction_mode import InteractionMode

    interaction_mode = InteractionMode()

    welcomer = None
    if config.BERKY_WELCOMER_ENABLED:
        if camera_worker is not None and not args.no_camera:
            # Deliberately does NOT reuse args.head_tracker/CameraWorker's own tracker:
            # that one drives real-time face-following head movement (look_at_image), which
            # Welcomer has no business turning on as a side effect - it only needs face
            # *detection* (get_all_faces), not head-steering. Give it its own instance.
            try:
                from berkie_reachy.tts import CommandTTS, speak_sync_through_robot
                from berkie_reachy.welcomer import Welcomer

                def _make_yolo_head_tracker() -> Any:
                    # Imported here, not at module load - BERKY_WELCOMER_ENABLED defaults
                    # to true everywhere so the toggle is always available, but YOLO
                    # (ultralytics/supervision) is an optional extra (pyproject.toml's
                    # yolo_vision), not part of the base install. Deferring both the
                    # import and construction until Welcomer mode is actually switched to
                    # means a device that only ever uses Community Assistant never needs
                    # YOLO at all - see welcomer.py's _resolve_head_tracker.
                    from berkie_reachy.vision.yolo_head_tracker import HeadTracker as YoloHeadTracker

                    return YoloHeadTracker()

                welcomer_tts = CommandTTS()
                welcomer = Welcomer(
                    camera_worker=camera_worker,
                    head_tracker_factory=_make_yolo_head_tracker,
                    speak=lambda text: speak_sync_through_robot(text, welcomer_tts, robot, movement_manager),
                    interaction_mode=interaction_mode,
                )
            except Exception:
                logger.exception("Failed to initialize Welcomer; continuing without it")
                welcomer = None
        else:
            logger.warning("BERKY_WELCOMER_ENABLED is set but the camera is disabled; Welcomer will not run.")

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    logger.info(
        "Connecting directly to BERKIE_LLM_ENGINE_BASE_URL=%s (production llm_engine - "
        "no local bootstrap)",
        config.BERKIE_LLM_ENGINE_BASE_URL,
    )

    use_berky_backend = bool(config.BERKIE_LLM_ENGINE_CONVERSATION_ID)
    if use_berky_backend:
        handler = BerkyLiveHandler(movement_manager=movement_manager, interaction_mode=interaction_mode)
    else:
        handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    stream_manager: gr.Blocks | LocalStream | None = None

    mode_radio = None
    if args.gradio:
        if use_berky_backend:
            stream_inputs: List[Any] = [chatbot]
            if welcomer is not None:
                # Only shown when Welcomer is actually running alongside the Community
                # Assistant - otherwise there's nothing to switch to.
                mode_radio = gr.Radio(
                    choices=["Community Assistant", "Welcomer"],
                    value="Community Assistant",
                    label="Interaction Mode",
                )
                stream_inputs.append(mode_radio)

            stream = Stream(
                handler=handler,
                mode="send-receive",
                modality="audio",
                additional_inputs=stream_inputs,
                additional_outputs=[chatbot],
                additional_outputs_handler=update_chatbot,
                ui_args={"title": "Talk with Berky"},
            )
        else:
            api_key_textbox = gr.Textbox(
                label="OPENAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
            )

            from berkie_reachy.gradio_personality import PersonalityUI

            personality_ui = PersonalityUI()
            personality_ui.create_components()

            stream = Stream(
                handler=handler,
                mode="send-receive",
                modality="audio",
                additional_inputs=[
                    chatbot,
                    api_key_textbox,
                    *personality_ui.additional_inputs_ordered(),
                ],
                additional_outputs=[chatbot],
                additional_outputs_handler=update_chatbot,
                ui_args={"title": "Talk with Reachy Mini"},
            )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        if not use_berky_backend:
            personality_ui.wire_events(handler, stream_manager)

        if mode_radio is not None:

            def _on_mode_change(selected: str) -> None:
                interaction_mode.mode = (
                    interaction_mode.WELCOMER if selected == "Welcomer" else interaction_mode.COMMUNITY_ASSISTANT
                )
                logger.info("Interaction mode switched to: %s", selected)

            with stream_manager:
                mode_radio.change(fn=_on_mode_change, inputs=[mode_radio], outputs=[])

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
            interaction_mode=interaction_mode if welcomer is not None else None,
            using_berky_backend=use_berky_backend,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()
    if welcomer:
        welcomer.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()
        if welcomer:
            welcomer.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class BerkieReachy(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = BerkieReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
