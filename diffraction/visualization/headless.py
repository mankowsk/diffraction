"""Headless browser for rendering SWIFT scenes without opening a window.

SWIFT renders its scene in a browser page (Three.js/WebGL). ``HeadlessBrowser`` drives a
headless Chrome via Playwright and implements the ``webbrowser`` controller interface, so it
can be registered with :func:`webbrowser.register` and passed to ``swift.Swift.launch(browser=...)``.
Screenshots are written directly to a file path instead of the browser's Downloads folder.

Playwright is an optional dependency, only needed for this module::

    conda install -c conda-forge playwright-python   # or: pip install playwright
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from diffraction.visualization.scene import VisualizationError


class HeadlessBrowser:
    """Headless Chrome page usable as a ``webbrowser`` controller for SWIFT.

    All Playwright calls run in one dedicated worker thread: Playwright's sync API cannot
    run inside an asyncio event loop, which Jupyter kernels always have, and its objects
    must be used from the thread that created them.

    Attributes:
        page: The Playwright page showing the SWIFT scene (use only via :meth:`call`)
    """

    def __init__(self, width: int = 1200, height: int = 900, channel: str | None = "chrome") -> None:
        """Start a headless browser.

        Args:
            width: Viewport width in pixels (default 1200)
            height: Viewport height in pixels (default 900)
            channel: Playwright browser channel; ``"chrome"`` uses the installed Google
                Chrome, ``None`` uses Playwright's own Chromium (``playwright install chromium``)

        Raises:
            VisualizationError: If Playwright is not installed or the browser cannot be started
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise VisualizationError(
                "Headless rendering requires Playwright. Install with:\n"
                "  conda install -c conda-forge playwright-python\n"
                "  or: pip install playwright"
            ) from e

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="headless-browser")

        def start():
            self._playwright = sync_playwright().start()
            try:
                self._browser = self._playwright.chromium.launch(channel=channel, headless=True)
            except Exception:
                self._playwright.stop()
                raise
            self.page = self._browser.new_page(viewport={"width": width, "height": height})

        try:
            self.call(start)
        except Exception as e:
            self._executor.shutdown()
            raise VisualizationError(f"Could not start headless browser (channel={channel!r}): {e}") from e

    def call(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run ``func(*args)`` in the browser thread and return its result."""
        return self._executor.submit(func, *args).result()

    def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
        """Load ``url`` in the headless page (``webbrowser`` controller interface)."""
        self.call(self.page.goto, url)
        return True

    def open_new(self, url: str) -> bool:
        return self.open(url)

    def open_new_tab(self, url: str) -> bool:
        return self.open(url)

    def screenshot(self, path: str | Path, wait: float = 0.5) -> Path:
        """Save the rendered SWIFT canvas as PNG.

        Args:
            path: Output file path
            wait: Time in seconds to let the scene render before capturing (default 0.5)

        Returns:
            Path: The output file path
        """
        path = Path(path)

        def capture():
            self.page.wait_for_timeout(wait * 1000)
            self.page.locator("canvas").first.screenshot(path=str(path))

        self.call(capture)
        return path

    def close(self) -> None:
        """Close the browser and stop Playwright."""

        def stop():
            self._browser.close()
            self._playwright.stop()

        self.call(stop)
        self._executor.shutdown()
