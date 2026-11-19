"""Jupyter controls (ipywidgets) for the lab-frame diffraction view."""

from typing import TYPE_CHECKING, Any

from diffraction.visualization.lab_view import LabVisualization

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

SLIDER_ANGLES = ("mu", "eta", "chi", "phi", "gamma", "delta")


def lab_view_widget(
    crystal: "Crystal",
    energy: float,
    hkl: tuple[int, int, int] | None = None,
    browser: str | None = "notebook",
    vis: LabVisualization | None = None,
) -> tuple[LabVisualization, Any]:
    """Lab view with sliders for the angles and a selector for ``calc_angles`` solutions.

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
    import ipywidgets as w

    if vis is None:
        vis = LabVisualization(crystal, energy).show(browser=browser)

    sliders = {
        name: w.FloatSlider(value=vis.angles[name], min=-180, max=180, step=0.01, description=name,
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

    def refresh_readout():
        readout.value = "<pre>" + vis.summary() + "</pre>"

    def on_slider(_change):
        if state["updating"]:
            return
        vis.set_angles({k: s.value for k, s in sliders.items()})
        refresh_readout()

    def set_sliders(angles):
        state["updating"] = True
        try:
            for name, slider in sliders.items():
                slider.value = float(angles[name])
        finally:
            state["updating"] = False

    def on_energy(_change):
        vis.set_angles(energy=energy_box.value)
        refresh_readout()

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
        vis.set_angles(row)
        refresh_readout()

    for slider in sliders.values():
        slider.observe(on_slider, names="value")
    energy_box.observe(on_energy, names="value")
    solve.on_click(on_solve)
    solutions.observe(on_solution, names="value")

    refresh_readout()
    if hkl:
        on_solve()

    controls = w.VBox([
        w.HBox([hkl_box, energy_box, solve]),
        solutions,
        *sliders.values(),
        readout,
    ])
    return vis, controls
