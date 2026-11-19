"""Shared SWIFT session handling: launching (incl. headless), snapshots and closing."""

import numpy as np


class VisualizationError(Exception):
    """Base exception for visualization-related errors."""

    pass


class SwiftScene:
    """Base class for visualizations rendered in a SWIFT environment.

    Attributes:
        env: SWIFT environment instance (None until launched)
    """

    #: Point the camera looks at in :meth:`snapshot` if ``look_at`` is not given
    default_look_at = np.zeros(3)

    def __init__(self) -> None:
        self.env = None
        self._headless = None

    def _launch(self, browser: str | None = None) -> None:
        """Start a SWIFT environment.

        Args:
            browser: Browser passed to SWIFT (e.g. ``"notebook"``, ``"firefox"``; default:
                system browser). ``"headless"`` renders in a headless Chrome without a
                window (requires Playwright); use :meth:`snapshot` to save images.
        """
        try:
            import swift
        except ImportError:
            raise VisualizationError(
                "SWIFT not installed. Install with:\n"
                "  pip install git+https://github.com/mankowsk/swift.git"
            )

        if browser == "headless":
            import webbrowser
            from diffraction.visualization.headless import HeadlessBrowser

            self._headless = HeadlessBrowser()
            webbrowser.register("diffraction-headless", None, self._headless)
            browser = "diffraction-headless"
        self.env = swift.Swift()
        self.env.launch(realtime=True, browser=browser)

    def snapshot(
        self,
        path: str,
        position: np.ndarray | list[float] | None = None,
        look_at: np.ndarray | list[float] | None = None,
        wait: float = 0.5,
    ) -> str:
        """Save a PNG of the scene rendered in headless mode (``show(browser="headless")``).

        Args:
            path: Output file path
            position: Camera position [x, y, z] in scene coordinates (optional)
            look_at: Point [x, y, z] the camera looks at (optional, used with ``position``)
            wait: Time in seconds to let the scene render before capturing (default 0.5)

        Returns:
            str: The output file path

        Raises:
            RuntimeError: If the visualization was not started in headless mode. In a
                browser session use ``env.screenshot()`` (saved to the Downloads folder).
        """
        if self._headless is None:
            raise RuntimeError(
                "snapshot() requires headless mode: call show(browser='headless'). "
                "In a browser session use vis.env.screenshot() instead."
            )
        if position is not None:
            self.env.set_camera_pose(position, look_at if look_at is not None else self.default_look_at)
        self.env.step(0)
        return str(self._headless.screenshot(path, wait=wait))

    def close(self) -> None:
        """Close the SWIFT session and, in headless mode, the headless browser."""
        if self.env is not None:
            self.env.close()
            self.env = None
        if self._headless is not None:
            self._headless.close()
            self._headless = None
