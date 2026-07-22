"""OpenMedVisionX application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "OpenMedVisionX GUI requires PyQt5. Install the base GUI dependencies first."
        ) from exc

    from .ui import OpenMedVisionXWindow
    from .ui.theme import APP_STYLE, application_font

    application = QApplication.instance()
    if application is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        rounding_policies = getattr(Qt, "HighDpiScaleFactorRoundingPolicy", None)
        pass_through = getattr(rounding_policies, "PassThrough", None)
        set_rounding_policy = getattr(
            QApplication,
            "setHighDpiScaleFactorRoundingPolicy",
            None,
        )
        if pass_through is not None and set_rounding_policy is not None:
            set_rounding_policy(pass_through)
        application = QApplication(sys.argv)
    application.setApplicationName("OpenMedVisionX")
    application.setApplicationDisplayName(
        "OpenMedVisionX: An Open Interactive Platform for Medical Computer Vision "
        "Learning and Exploration"
    )
    application.setOrganizationName("OpenMedVisionX contributors")
    application.setFont(application_font())
    application.setStyleSheet(APP_STYLE)
    window = OpenMedVisionXWindow()
    window.show()
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
