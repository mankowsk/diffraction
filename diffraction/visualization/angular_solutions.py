"""Angular solutions visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_angular_solutions(crystal, hkl: tuple[int, int, int], energy: float) -> tuple:
    """Plot all possible angular solutions for a given reflection.

    Creates a comprehensive figure with multiple subplots showing:
    - All angle solutions (mu, delta, gamma/nu, eta, chi, phi) in a table format
    - Bar chart comparing angle values across different solutions
    - Polar plot of the scattering geometry

    Coordinate System Convention:
        - Incident x-ray beam along the positive y axis
        - All angles = 0: sample surface normal along the z axis
        - All angles = 0: scattered beam along the positive y axis

    Args:
        crystal: Crystal object with constraints set
        hkl: Miller indices (h, k, l)
        energy: X-ray energy in eV

    Returns:
        tuple: (Figure, dict of Axes objects) where axes contains:
            - 'table': Angle values table
            - 'bar': Bar chart comparison
            - 'polar': Polar scattering plot

    Example:
        >>> fig, axes = plot_angular_solutions(crystal, (0, 0, 1), 8048.0)
    """
    angles_df = crystal.calc_angles(hkl, energy, pandas=True)

    fig = plt.figure(figsize=(14, 10))
    fig.set_layout_engine('tight')
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])

    axes = {}

    # Top row: Table of angle values
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    table_data = angles_df.round(2).values.tolist()
    columns = angles_df.keys()
    #columns = [f'Sol {i}' for i in range(len(angles_df.columns))]
    table = ax_table.table(
        cellText=table_data,
        #colLabels=[f'{col}\n{angles_df.index[i]}' for i, col in enumerate(columns)],
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(columns))))
    ax_table.set_title(f'Angular Solutions for hkl={hkl} at {energy} eV', pad=20)
    axes['table'] = ax_table

    # Second row: Bar chart of angles
    ax_bar = fig.add_subplot(gs[1, 0])
    angle_names = ['mu', 'delta', 'gamma (nu)', 'eta', 'chi', 'phi']
    valid_angles = [name for name in angle_names if name in angles_df.index]
    x = np.arange(len(valid_angles))
    width = 0.8

    for i, col_name in enumerate(angles_df.columns):
        values = [angles_df.loc[angle, col_name] if angle in angles_df.index else np.nan
                  for angle in valid_angles]
        ax_bar.bar(x + i * width - width * (len(angles_df.columns) - 1) / 2,
                   values, width=width, label=col_name)

    ax_bar.set_ylabel('Angle (degrees)')
    ax_bar.set_title('Angular Solutions Comparison')
    ax_bar.set_xticks(x + width * (len(angles_df.columns) - 1) / 2)
    ax_bar.set_xticklabels(valid_angles, rotation=45)
    ax_bar.legend(loc='best', fontsize=8)
    ax_bar.grid(axis='y', alpha=0.3)
    axes['bar'] = ax_bar

    # Second row: Polar plot of scattering geometry
    ax_polar = fig.add_subplot(gs[1, 1], projection='polar')
    for i, col_name in enumerate(angles_df.columns):
        angles_rad = []
        radii = []
        for angle_name in ['mu', 'delta', 'gamma (nu)', 'eta', 'chi', 'phi']:
            if angle_name in angles_df.index:
                val = angles_df.loc[angle_name, col_name]
                angles_rad.append(val)
                radii.append(i + 1)

        if len(angles_rad) > 0:
            ax_polar.plot(angles_rad, radii, 'o-', label=col_name, linewidth=2, markersize=6)

    ax_polar.set_title('Scattering Geometry', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    axes['polar'] = ax_polar

    # Bottom row: Summary statistics
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    summary_text = ''
    for col_name in angles_df.columns:
        summary_text += f'\n{col_name}:\n'
        for angle_name in valid_angles:
            if angle_name in angles_df.index:
                val = angles_df.loc[angle_name, col_name]
                summary_text += f'  {angle_name}: {val:.2f}°\n'

    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Angular Solutions for {crystal.name} - Reflection ({hkl[0]}, {hkl[1]}, {hkl[2]}) '
                 f'at {energy} eV X-ray energy', fontsize=12, y=0.98)

    return fig, axes


def plot_angle_solutions_scatter(crystal, hkl: tuple[int, int, int], energy: float) -> tuple:
    """Plot angular solutions as scatter points in angle space.

    Creates a pairplot-style visualization showing relationships between
    different diffractometer angles across all solutions.

    Coordinate System Convention:
        - Incident x-ray beam along the positive y axis
        - All angles = 0: sample surface normal along the z axis
        - All angles = 0: scattered beam along the positive y axis

    Args:
        crystal: Crystal object with constraints set
        hkl: Miller indices (h, k, l)
        energy: X-ray energy in eV

    Returns:
        tuple: (Figure, Axes objects) - Figure and dictionary of scatter plots
    """
    angles_df = crystal.calc_angles(hkl, energy, pandas=True)

    angle_names = ['mu', 'delta', 'gamma', 'eta', 'chi', 'phi']
    valid_angles = [name for name in angle_names if name in angles_df.index]

    n = len(valid_angles)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    fig.set_layout_engine('tight')
    if n == 1:
        axes = np.array([[axes]])
    elif n == 2:
        axes = np.expand_dims(axes, 0) if axes.ndim == 1 else axes

    for i, angle_x in enumerate(valid_angles):
        for j, angle_y in enumerate(valid_angles):
            ax = axes[i, j] if n > 1 else axes

            if i == j:
                # Diagonal: histogram of single angle
                all_values = [angles_df.loc[angle_x, col] for col in angles_df.columns]
                ax.hist(all_values, bins=min(10, len(all_values)), alpha=0.7, edgecolor='black')
                ax.set_title(f'{angle_y}\n{angle_x}')
                ax.set_xlabel(angle_x if i == n - 1 else '')
                ax.set_ylabel(angle_y if j == 0 else '')
            else:
                # Off-diagonal: scatter plot
                for col in angles_df.columns:
                    x_vals = [angles_df.loc[angle_x, col]]
                    y_vals = [angles_df.loc[angle_y, col]]
                    ax.scatter(x_vals, y_vals, s=100, alpha=0.7, label=col)

                ax.set_xlabel(angle_x, rotation=45)
                ax.set_ylabel(angle_y)
                if j == 0:
                    ax.set_ylabel(angle_y)

    fig.suptitle(f'Angle Correlation Scatter Plot for hkl={hkl}', fontsize=12, y=0.98)
    fig.set_layout_engine('tight')

    return fig, axes


def plot_constraint_sensitivity(crystal, hkl: tuple[int, int, int], energy: float, solution: int, constraint_name: str, range_min: float,
                                range_max: float, n_points: int = 20, angles_to_plot: list = ['mu', 'delta', 'gamma', 'eta', 'chi', 'phi']) -> tuple:
    """Plot how angular solutions vary with a specific constraint.

    Performs a parameter sweep on the specified constraint and shows
    how each diffractometer angle responds to changes.

    Args:
        crystal: Crystal object with constraints set
        hkl: Miller indices (h, k, l)
        energy: X-ray energy in eV
        solution: which of the four scattering geometries to plot (integer out of 0, 1, 2, 4)
        constraint_name: Name of the constraint to vary (e.g., 'mu', 'eta')
        range_min: Minimum value for the constraint sweep
        range_max: Maximum value for the constraint sweep
        n_points: Number of points in the sweep
        angles_to_plot: list of angles to plot out of ['mu', 'delta', 'gamma', 'eta', 'chi', 'phi']

    Returns:
        tuple: (Figure, Axes, x_values, results) - Figure and line plot showing sensitivity
    """
    original_constraint = crystal.constraints.get(constraint_name)
    x_values = np.linspace(range_min, range_max, n_points)
    results = {angle: {f"sol. {n}": [] for n in range(4)} for angle in angles_to_plot}

    for val in x_values:
        crystal.constraints[constraint_name] = val

        try:
            angles = crystal.calc_angles(hkl, energy, pandas=True)
            for angle in angles_to_plot:
                for idx in angles.index:
                    results[angle][idx].append(angles[angle][idx])
        except Exception as e:
            print(f"Warning: Failed to calculate at {constraint_name}={val}: {e}")
            for angle in angles_to_plot:
                for k in results[angle].keys():
                    results[angle][k].append(np.nan)
            continue

    if original_constraint is not None:
        crystal.constraints[constraint_name] = original_constraint

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_layout_engine('tight')

    colors = plt.cm.tab10(range(len(angles_to_plot)))
    for i, angle in enumerate(angles_to_plot):
        ax.plot(x_values, results[angle][f"sol. {solution}"], 'o-', color=colors[i],
                label=angle, markersize=4, linewidth=1.5)

    ax.set_xlabel(f'{constraint_name} (degrees)', fontsize=12)
    ax.set_ylabel('Diffractometer Angle (degrees)', fontsize=12)
    ax.set_title(f'Constraint Sensitivity: {constraint_name}', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig, ax, x_values, results
