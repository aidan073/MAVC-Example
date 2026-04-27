"""Viewport chrome and resolution presets for demo / recording runs.

`SimulationApp(..., hide_ui=True)` removes most *application-level* UI (main menus,
docks, outliner, etc.). The viewport still has its own HUD, axis guides, extension
menubars (camera/display/… tabs), and dock tab chrome. Those are controlled here via
carb settings and the viewport window — they are not redundant with hide_ui.

:func:`apply_demo_viewport_visuals` also sets the Kit main window to fullscreen.
"""

import carb
import omni.appwindow


def apply_demo_viewport_visuals(vp_api, vp_window=None) -> None:
    """Strip viewport-local HUD/menubars/overlays and set a high RT viewport resolution.

    Call this even when ``hide_ui=True`` on ``SimulationApp``: that flag does not turn
    off viewport-internal chrome.
    """
    app_window = omni.appwindow.get_default_app_window()
    if app_window is not None:
        app_window.set_fullscreen(True)
