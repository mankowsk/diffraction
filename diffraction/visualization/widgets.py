"""Jupyter controls (ipywidgets) for the lab-frame and reciprocal-space views."""

from typing import TYPE_CHECKING, Any, Sequence

from diffraction.visualization.diffraction_scene import DiffractionScene
from diffraction.visualization.lab_view import LabVisualization
from diffraction.visualization.reciprocal_view import ReciprocalVisualization

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

SLIDER_ANGLES = ("mu", "eta", "chi", "phi", "gamma", "delta")


def angle_controls(
    crystal: "Crystal",
    energy: float,
    views: DiffractionScene | Sequence[DiffractionScene],
    hkl: tuple[int, int, int] | None = None,
) -> Any:
    """Sliders for the angles, energy and a selector for ``calc_angles`` solutions.

    All ``views`` (e.g. a lab view and a reciprocal-space view) move together.

    Args:
        crystal: Crystal with a UB matrix
        energy: X-ray energy in eV
        views: One or several views (``LabVisualization``, ``ReciprocalVisualization``)
        hkl: Reflection whose ``calc_angles`` solutions are offered initially (optional)

    Returns:
        ipywidgets.VBox with the controls
    """
    import ipywidgets as w

    views = [views] if isinstance(views, DiffractionScene) else list(views)
    # The reciprocal view's summary includes the reflections in diffraction condition
    reporter = next((v for v in views if isinstance(v, ReciprocalVisualization)), views[0])

    sliders = {
        name: w.FloatSlider(value=views[0].angles[name], min=-180, max=180, step=0.01, description=name,
                            continuous_update=False, readout_format=".2f", layout=w.Layout(width="420px"))
        for name in SLIDER_ANGLES
    }
    energy_box = w.BoundedFloatText(value=energy, min=100, max=100000, step=10, description="E (eV)")
    hkl_box = w.Text(value=" ".join(str(v) for v in hkl) if hkl else "", description="hkl",
                     placeholder="h k l", layout=w.Layout(width="200px"))
    solve = w.Button(description="calc_angles")
    solutions = w.Dropdown(options=[], description="solution", layout=w.Layout(width="420px"))
    readout = w.HTML()
    state = {"updating": False, "rows": []}

    def set_all(angles=None, **kwargs):
        for view in views:
            view.set_angles(angles, **kwargs)
        readout.value = "<pre>" + reporter.summary() + "</pre>"

    def on_slider(_change):
        if not state["updating"]:
            set_all({k: s.value for k, s in sliders.items()})

    def set_sliders(angles):
        state["updating"] = True
        try:
            for name, slider in sliders.items():
                slider.value = float(angles[name])
        finally:
            state["updating"] = False

    def on_energy(_change):
        set_all(energy=energy_box.value)

    def on_solve(_button=None):
        try:
            h, k, l = (int(v) for v in hkl_box.value.replace(",", " ").split())
            df = crystal.calc_angles((h, k, l), energy_box.value)
        except Exception as e:  # invalid input or no solution
            solutions.options = []
            readout.value = f"<pre>calc_angles failed: {e}</pre>"
            return
        state["rows"] = [row for _, row in df.iterrows()]
        solutions.options = [
            (", ".join(f"{a}={row[a]:.2f}" for a in SLIDER_ANGLES), i) for i, row in enumerate(state["rows"])
        ]
        if state["rows"]:
            solutions.value = 0
            on_solution({"new": 0})

    def on_solution(change):
        index = change["new"]
        if index is None or index >= len(state["rows"]):
            return
        row = state["rows"][index]
        set_sliders(row)
        set_all(row)

    for slider in sliders.values():
        slider.observe(on_slider, names="value")
    energy_box.observe(on_energy, names="value")
    solve.on_click(on_solve)
    solutions.observe(on_solution, names="value")

    readout.value = "<pre>" + reporter.summary() + "</pre>"
    if hkl:
        on_solve()

    return w.VBox([w.HBox([hkl_box, energy_box, solve]), solutions, *sliders.values(), readout])


def lab_view_widget(
    crystal: "Crystal",
    energy: float,
    hkl: tuple[int, int, int] | None = None,
    browser: str | None = "notebook",
    vis: LabVisualization | None = None,
) -> tuple[LabVisualization, Any]:
    """Lab view with angle controls (see :func:`angle_controls`).

    In a notebook cell::

        vis, controls = lab_view_widget(crystal, 7500, hkl=(0, 0, 15))
        controls

    Args:
        crystal: Crystal with a UB matrix
        energy: X-ray energy in eV
        hkl: Reflection whose ``calc_angles`` solutions are offered initially (optional)
        browser: SWIFT browser (default ``"notebook"``: scene embedded in the cell output)
        vis: Existing :class:`LabVisualization` to control instead of creating one

    Returns:
        tuple: (LabVisualization, ipywidgets.VBox with the controls)
    """
    if vis is None:
        vis = LabVisualization(crystal, energy).show(browser=browser)
    return vis, angle_controls(crystal, energy, vis, hkl)


def reciprocal_view_widget(
    crystal: "Crystal",
    energy: float,
    hkl: tuple[int, int, int] | None = None,
    browser: str | None = "notebook",
    vis: ReciprocalVisualization | None = None,
) -> tuple[ReciprocalVisualization, Any]:
    """Reciprocal-space (Ewald) view with angle controls (see :func:`angle_controls`).

    Args:
        crystal: Crystal with a UB matrix
        energy: X-ray energy in eV
        hkl: Reflection whose ``calc_angles`` solutions are offered initially (optional)
        browser: SWIFT browser (default ``"notebook"``)
        vis: Existing :class:`ReciprocalVisualization` to control instead of creating one

    Returns:
        tuple: (ReciprocalVisualization, ipywidgets.VBox with the controls)
    """
    if vis is None:
        vis = ReciprocalVisualization(crystal, energy).show(browser=browser)
    return vis, angle_controls(crystal, energy, vis, hkl)
