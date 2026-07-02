from __future__ import annotations

import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Settings
# ============================================================

MODES_TO_RENDER = ["kepler_2d", "kepler_3d", "nbody"]
# For a single output, set e.g. ["kepler_3d"]
# For all three, keep the default list above.

DAYS_PER_YEAR = 365.25
J2000_JD = 2451545.0

SUN_COLOR = "#ffcf6b"
BG = "#050816"

plt.style.use("dark_background")


# ============================================================
# Data
# JPL approximate-position elements from Table 1, with
# masses added for the N-body simulation.
# ============================================================

@dataclass(frozen=True)
class Body:
    name: str
    color: str
    mass_solar: float

    a0: float
    a1: float
    e0: float
    e1: float
    I0: float
    I1: float
    L0: float
    L1: float
    long_peri0: float
    long_peri1: float
    long_node0: float
    long_node1: float


PLANETS: dict[str, Body] = {
    "Mercury": Body(
        "Mercury", "#b7b7b7", 1.660120e-7,
        0.38709927, 0.00000037,
        0.20563593, 0.00001906,
        7.00497902, -0.00594749,
        252.25032350, 149472.67411175,
        77.45779628, 0.16047689,
        48.33076593, -0.12534081,
    ),
    "Venus": Body(
        "Venus", "#f7c948", 2.4478383e-6,
        0.72333566, 0.00000390,
        0.00677672, -0.00004107,
        3.39467605, -0.00078890,
        181.97909950, 58517.81538729,
        131.60246718, 0.00268329,
        76.67984255, -0.27769418,
    ),
    # JPL Table 1 row is Earth/Moon barycenter; we use it as "Earth" for visualization.
    "Earth": Body(
        "Earth", "#4cc9f0", 3.0034896e-6,
        1.00000261, 0.00000562,
        0.01671123, -0.00004392,
        -0.00001531, -0.01294668,
        100.46457166, 35999.37244981,
        102.93768193, 0.32327364,
        0.0, 0.0,
    ),
    "Mars": Body(
        "Mars", "#ff6b6b", 3.227151e-7,
        1.52371034, 0.00001847,
        0.09339410, 0.00007882,
        1.84969142, -0.00813131,
        -4.55343205, 19140.30268499,
        -23.94362959, 0.44441088,
        49.55953891, -0.29257343,
    ),
    "Jupiter": Body(
        "Jupiter", "#ff9f1c", 9.545942e-4,
        5.20288700, -0.00011607,
        0.04838624, -0.00013253,
        1.30439695, -0.00183714,
        34.39644051, 3034.74612775,
        14.72847983, 0.21252668,
        100.47390909, 0.20469106,
    ),
    "Saturn": Body(
        "Saturn", "#ffd166", 2.858150e-4,
        9.53667594, -0.00125060,
        0.05386179, -0.00050991,
        2.48599187, 0.00193609,
        49.95424423, 1222.49362201,
        92.59887831, -0.41897216,
        113.66242448, -0.28867794,
    ),
    "Uranus": Body(
        "Uranus", "#7bdff2", 4.366244e-5,
        19.18916464, -0.00196176,
        0.04725744, -0.00004397,
        0.77263783, -0.00242939,
        313.23810451, 428.48202785,
        170.95427630, 0.40805281,
        74.01692503, 0.04240589,
    ),
    "Neptune": Body(
        "Neptune", "#5e60ce", 5.151389e-5,
        30.06992276, 0.00026291,
        0.00859048, 0.00005105,
        1.77004347, 0.00035372,
        -55.12002969, 218.45945325,
        44.96476227, -0.32241464,
        131.78422574, -0.00508664,
    ),
}

ALL_PLANETS = list(PLANETS.keys())
INNER_PLANETS = ["Mercury", "Venus", "Earth", "Mars"]


DISPLAY_SIZES_2D = {
    "Mercury": 22,
    "Venus": 38,
    "Earth": 44,
    "Mars": 36,
    "Jupiter": 62,
    "Saturn": 58,
    "Uranus": 52,
    "Neptune": 52,
    "Sun": 160,
}

DISPLAY_SIZES_3D = {
    "Mercury": 4.0,
    "Venus": 4.6,
    "Earth": 5.0,
    "Mars": 4.4,
    "Jupiter": 6.0,
    "Saturn": 5.8,
    "Uranus": 5.4,
    "Neptune": 5.4,
    "Sun": 10.0,
}


# ============================================================
# Utilities
# ============================================================

def find_desktop() -> Path:
    candidates = [
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path.cwd()


def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def wrap_rad(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def solve_kepler_rad(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    M = wrap_rad(M)
    E = M if e < 0.8 else math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return E


def current_elements(body: Body, t_days: float) -> dict[str, float]:
    T = t_days / 36525.0  # centuries from J2000.0

    a = body.a0 + body.a1 * T
    e = body.e0 + body.e1 * T
    I_deg = body.I0 + body.I1 * T
    L_deg = body.L0 + body.L1 * T
    varpi_deg = body.long_peri0 + body.long_peri1 * T
    Omega_deg = body.long_node0 + body.long_node1 * T
    omega_deg = wrap_deg(varpi_deg - Omega_deg)

    return {
        "a": a,
        "e": e,
        "I_deg": I_deg,
        "L_deg": L_deg,
        "varpi_deg": varpi_deg,
        "Omega_deg": Omega_deg,
        "omega_deg": omega_deg,
    }


def orbital_rotation_matrix(I_deg: float, Omega_deg: float, omega_deg: float) -> np.ndarray:
    I = math.radians(I_deg)
    Omega = math.radians(Omega_deg)
    omega = math.radians(omega_deg)

    cO, sO = math.cos(Omega), math.sin(Omega)
    ci, si = math.cos(I), math.sin(I)
    cw, sw = math.cos(omega), math.sin(omega)

    # Maps [x', y'] in the orbital plane to ecliptic coordinates.
    return np.array([
        [cw * cO - sw * sO * ci, -sw * cO - cw * sO * ci],
        [cw * sO + sw * cO * ci, -sw * sO + cw * cO * ci],
        [sw * si,                cw * si],
    ])


def state_from_elements(body: Body, t_days: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Heliocentric ecliptic position and velocity in AU and AU/year.
    """
    el = current_elements(body, t_days)
    a = el["a"]
    e = max(0.0, min(el["e"], 0.999999999))
    I_deg = el["I_deg"]
    Omega_deg = el["Omega_deg"]
    omega_deg = el["omega_deg"]
    L_deg = el["L_deg"]
    varpi_deg = el["varpi_deg"]

    M_deg = wrap_deg(L_deg - varpi_deg)
    M = math.radians(M_deg)

    E = solve_kepler_rad(M, e)

    cosE, sinE = math.cos(E), math.sin(E)
    one_minus_e_cosE = 1.0 - e * cosE

    x_orb = a * (cosE - e)
    y_orb = a * math.sqrt(1.0 - e * e) * sinE

    # Time unit is years, distance is AU. G in these units is 4*pi^2.
    mu = 4.0 * math.pi**2 * (1.0 + body.mass_solar)
    n = math.sqrt(mu / (a**3))  # rad/year
    xdot_orb = -a * n * sinE / one_minus_e_cosE
    ydot_orb = a * n * math.sqrt(1.0 - e * e) * cosE / one_minus_e_cosE

    R = orbital_rotation_matrix(I_deg, Omega_deg, omega_deg)
    pos = R @ np.array([x_orb, y_orb])
    vel = R @ np.array([xdot_orb, ydot_orb])

    return pos, vel


def orbit_curve(body: Body, n: int = 720) -> np.ndarray:
    """
    Returns a 3xN array of orbit points in ecliptic coordinates.
    """
    el = current_elements(body, 0.0)
    a = el["a"]
    e = max(0.0, min(el["e"], 0.999999999))
    I_deg = el["I_deg"]
    Omega_deg = el["Omega_deg"]
    omega_deg = el["omega_deg"]

    E = np.linspace(0.0, 2.0 * math.pi, n)
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1.0 - e * e) * np.sin(E)

    R = orbital_rotation_matrix(I_deg, Omega_deg, omega_deg)
    pts = R @ np.vstack([x_orb, y_orb])
    return pts


def make_starfield(ax, limit: float, n: int = 350) -> None:
    rng = np.random.default_rng(7)
    sx = rng.uniform(-limit, limit, n)
    sy = rng.uniform(-limit, limit, n)
    ss = rng.uniform(2, 14, n)
    ax.scatter(sx, sy, s=ss, c="white", alpha=0.14, linewidths=0, zorder=0)


def save_animation(anim, output_dir: Path, basename: str, fps: int = 30, dpi: int = 150) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = output_dir / f"{basename}.mp4"
    gif_path = output_dir / f"{basename}.gif"

    if shutil.which("ffmpeg") is not None:
        try:
            anim.save(mp4_path, writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=dpi)
            print(f"Saved: {mp4_path}")
            return
        except Exception as exc:
            print(f"MP4 save failed: {exc}. Falling back to GIF...")

    anim.save(gif_path, writer=PillowWriter(fps=max(12, fps // 2)), dpi=dpi)
    print(f"Saved: {gif_path}")


def compute_acceleration(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    N-body acceleration with G = 4*pi^2 in AU^3 / (yr^2 * solar_mass).
    positions shape: (N, 3)
    masses shape: (N,)
    """
    G = 4.0 * math.pi**2
    delta = positions[None, :, :] - positions[:, None, :]  # r_j - r_i
    dist2 = np.sum(delta * delta, axis=2) + np.eye(len(masses))
    inv_dist3 = 1.0 / np.power(dist2, 1.5)
    np.fill_diagonal(inv_dist3, 0.0)
    acc = G * np.sum(delta * inv_dist3[:, :, None] * masses[None, :, None], axis=1)
    return acc


def velocity_verlet_step(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, dt_years: float):
    a0 = compute_acceleration(positions, masses)
    v_half = velocities + 0.5 * dt_years * a0
    r_new = positions + dt_years * v_half
    a1 = compute_acceleration(r_new, masses)
    v_new = v_half + 0.5 * dt_years * a1
    return r_new, v_new


# ============================================================
# 2D Kepler animation
# ============================================================

def render_kepler_2d(output_dir: Path) -> None:
    bodies = INNER_PLANETS
    total_days = 4.0 * DAYS_PER_YEAR
    days_per_frame = 2.0
    frames = int(total_days / days_per_frame) + 1

    limit = max((PLANETS[name].a0 * (1.0 + PLANETS[name].e0) for name in bodies)) * 1.35

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Keplerian Planetary Motion — 2D", fontsize=18, pad=16, color="white")

    make_starfield(ax, limit, n=300)

    # Sun glow
    for size, alpha in [(2200, 0.06), (900, 0.15), (350, 0.95)]:
        ax.scatter([0], [0], s=size, c=SUN_COLOR, alpha=alpha, edgecolors="none", zorder=10)

    trail_len = 140
    trail_x: list[deque[float]] = []
    trail_y: list[deque[float]] = []
    trails = []
    markers = []
    labels = []

    for name in bodies:
        body = PLANETS[name]
        curve = orbit_curve(body)
        ax.plot(curve[0], curve[1], color=body.color, lw=1.15, alpha=0.35, zorder=1)

        trail, = ax.plot([], [], color=body.color, lw=2.0, alpha=0.95, zorder=3)
        marker = ax.scatter([0], [0], s=DISPLAY_SIZES_2D[name], c=body.color,
                            edgecolors="white", linewidths=0.7, zorder=4)
        label = ax.text(0, 0, name, fontsize=9.5, color="white",
                        ha="left", va="bottom", zorder=5)

        trails.append(trail)
        markers.append(marker)
        labels.append(label)
        trail_x.append(deque(maxlen=trail_len))
        trail_y.append(deque(maxlen=trail_len))

    earth_idx = bodies.index("Earth")

    earth_wedge = Polygon([[0, 0], [0, 0], [0, 0]],
                          closed=True, facecolor=PLANETS["Earth"].color,
                          edgecolor=PLANETS["Earth"].color, alpha=0.20, zorder=2)
    ax.add_patch(earth_wedge)

    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, color="white",
        bbox=dict(boxstyle="round,pad=0.45", fc=(0, 0, 0, 0.45), ec="none"),
        zorder=20,
    )

    footer = ax.text(
        0.5, 0.02,
        r"Kepler I: ellipses | Kepler II: equal areas in equal times | Kepler III: \(P^2 \propto a^3\)",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, color="#d8e1ff", alpha=0.9, zorder=20,
    )

    def update(frame: int):
        t_days = frame * days_per_frame
        positions = [state_from_elements(PLANETS[name], t_days)[0] for name in bodies]

        for i, (name, pos) in enumerate(zip(bodies, positions)):
            x, y = pos[0], pos[1]
            trail_x[i].append(x)
            trail_y[i].append(y)

            trails[i].set_data(list(trail_x[i]), list(trail_y[i]))
            markers[i].set_offsets(np.array([[x, y]]))
            labels[i].set_position((x + 0.05, y + 0.05))

        ex, ey = positions[earth_idx][0], positions[earth_idx][1]
        if len(trail_x[earth_idx]) >= 2:
            px = trail_x[earth_idx][-2]
            py = trail_y[earth_idx][-2]
        else:
            px, py = ex, ey

        earth_wedge.set_xy(np.array([[0, 0], [px, py], [ex, ey], [0, 0]]))
        swept_area = 0.5 * abs(px * ey - py * ex)

        info.set_text(
            f"Day: {t_days:6.1f}\n"
            f"Earth swept area over {days_per_frame:g} days: {swept_area:.6f} AU²\n"
            f"Areal speed ≈ {swept_area / days_per_frame:.6f} AU²/day"
        )

        return trails + markers + labels + [earth_wedge, info, footer]

    anim = FuncAnimation(fig, update, frames=frames, interval=33, blit=False)
    save_animation(anim, output_dir, "planetary_motion_kepler_2d", fps=30, dpi=160)
    plt.close(fig)


# ============================================================
# 3D Kepler animation
# ============================================================

def render_kepler_3d(output_dir: Path) -> None:
    bodies = ALL_PLANETS
    total_days = 8.0 * DAYS_PER_YEAR
    days_per_frame = 4.0
    frames = int(total_days / days_per_frame) + 1

    limit = max((PLANETS[name].a0 * (1.0 + PLANETS[name].e0) for name in bodies)) * 1.18

    fig = plt.figure(figsize=(11, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.set_title("Keplerian Planetary Motion — 3D Tilted Orbits", fontsize=18, pad=18, color="white")
    ax.view_init(elev=24, azim=40)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit * 0.18, limit * 0.18)
    try:
        ax.set_box_aspect((1, 1, 0.36))
    except Exception:
        pass

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # A faint ecliptic plane grid
    grid = np.linspace(-limit, limit, 2)
    for g in np.linspace(-limit, limit, 5):
        ax.plot(grid, np.full_like(grid, g), np.zeros_like(grid), color="white", alpha=0.04, lw=0.6)
        ax.plot(np.full_like(grid, g), grid, np.zeros_like(grid), color="white", alpha=0.04, lw=0.6)

    # Sun
    sun_marker, = ax.plot([0], [0], [0], marker="o", markersize=DISPLAY_SIZES_3D["Sun"] / 2,
                          color=SUN_COLOR, lw=0, zorder=10)

    trail_len = 160
    trail_x: list[deque[float]] = []
    trail_y: list[deque[float]] = []
    trail_z: list[deque[float]] = []
    trails = []
    markers = []

    for name in bodies:
        body = PLANETS[name]
        curve = orbit_curve(body, n=700)
        ax.plot(curve[0], curve[1], curve[2], color=body.color, lw=1.0, alpha=0.35)

        trail, = ax.plot([], [], [], color=body.color, lw=2.0, alpha=0.95)
        marker, = ax.plot([], [], [], marker="o", linestyle="None",
                          markersize=DISPLAY_SIZES_3D[name], color=body.color)

        trails.append(trail)
        markers.append(marker)
        trail_x.append(deque(maxlen=trail_len))
        trail_y.append(deque(maxlen=trail_len))
        trail_z.append(deque(maxlen=trail_len))

    legend_handles = [
        Line2D([0], [0], color=PLANETS[name].color, lw=2, label=name)
        for name in bodies
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
              framealpha=0.22, facecolor="black", edgecolor="none")

    info = ax.text2D(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.5, color="white",
        bbox=dict(boxstyle="round,pad=0.45", fc=(0, 0, 0, 0.45), ec="none"),
    )

    footer = ax.text2D(
        0.5, 0.02,
        "JPL orbital elements + 3D plane tilts",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, color="#d8e1ff", alpha=0.9,
    )

    def update(frame: int):
        t_days = frame * days_per_frame
        positions = [state_from_elements(PLANETS[name], t_days)[0] for name in bodies]

        for i, pos in enumerate(positions):
            x, y, z = pos[0], pos[1], pos[2]
            trail_x[i].append(x)
            trail_y[i].append(y)
            trail_z[i].append(z)

            trails[i].set_data(list(trail_x[i]), list(trail_y[i]))
            trails[i].set_3d_properties(list(trail_z[i]))

            markers[i].set_data([x], [y])
            markers[i].set_3d_properties([z])

        info.set_text(
            f"Day: {t_days:6.1f}\n"
            f"3D tilted orbits from JPL approximate elements\n"
            f"Current view is heliocentric ecliptic space"
        )

        return [sun_marker] + trails + markers + [info, footer]

    anim = FuncAnimation(fig, update, frames=frames, interval=33, blit=False)
    save_animation(anim, output_dir, "planetary_motion_kepler_3d", fps=30, dpi=160)
    plt.close(fig)


# ============================================================
# Newtonian gravity simulation
# ============================================================

def render_nbody(output_dir: Path) -> None:
    bodies = INNER_PLANETS
    total_days = 4.0 * DAYS_PER_YEAR
    days_per_frame = 2.0
    substeps = 10
    frames = int(total_days / days_per_frame) + 1

    limit = max((PLANETS[name].a0 * (1.0 + PLANETS[name].e0) for name in bodies)) * 1.40

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Newtonian Gravity Simulation — N-body", fontsize=18, pad=16, color="white")

    make_starfield(ax, limit, n=280)

    # Build initial heliocentric states at J2000.
    planet_positions = []
    planet_velocities = []
    masses = [1.0]  # Sun
    body_names = ["Sun"] + bodies

    for name in bodies:
        pos, vel = state_from_elements(PLANETS[name], 0.0)
        planet_positions.append(pos)
        planet_velocities.append(vel)
        masses.append(PLANETS[name].mass_solar)

    planet_positions = np.array(planet_positions, dtype=float)
    planet_velocities = np.array(planet_velocities, dtype=float)
    masses = np.array(masses, dtype=float)

    # Put the barycenter at the origin and total momentum at zero.
    sun_pos = -np.sum(planet_positions * masses[1:, None], axis=0) / masses[0]
    sun_vel = -np.sum(planet_velocities * masses[1:, None], axis=0) / masses[0]

    positions = np.vstack([sun_pos[None, :], planet_positions])
    velocities = np.vstack([sun_vel[None, :], planet_velocities])

    trail_len = 150
    trail_x: list[deque[float]] = []
    trail_y: list[deque[float]] = []
    trails = []
    markers = []
    labels = []

    for name in body_names:
        color = SUN_COLOR if name == "Sun" else PLANETS[name].color
        trail, = ax.plot([], [], color=color, lw=2.2 if name == "Sun" else 1.8,
                         alpha=0.95 if name == "Sun" else 0.90, zorder=3)
        marker = ax.scatter([0], [0], s=DISPLAY_SIZES_2D[name], c=color,
                            edgecolors="white" if name != "Sun" else "none",
                            linewidths=0.7 if name != "Sun" else 0.0, zorder=4)
        label = ax.text(0, 0, name, fontsize=10 if name == "Sun" else 9,
                        color="white", ha="left", va="bottom", zorder=5)

        trails.append(trail)
        markers.append(marker)
        labels.append(label)
        trail_x.append(deque(maxlen=trail_len))
        trail_y.append(deque(maxlen=trail_len))

    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, color="white",
        bbox=dict(boxstyle="round,pad=0.45", fc=(0, 0, 0, 0.45), ec="none"),
        zorder=20,
    )

    footer = ax.text(
        0.5, 0.02,
        "Pairwise Newtonian gravity with velocity Verlet integration",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, color="#d8e1ff", alpha=0.9, zorder=20,
    )

    dt_years = (days_per_frame / DAYS_PER_YEAR) / substeps

    def update(frame: int):
        nonlocal positions, velocities

        for _ in range(substeps):
            positions, velocities = velocity_verlet_step(positions, velocities, masses, dt_years)

        t_days = frame * days_per_frame
        com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
        plot_pos = positions - com

        for i, pos in enumerate(plot_pos):
            x, y = pos[0], pos[1]
            trail_x[i].append(x)
            trail_y[i].append(y)

            trails[i].set_data(list(trail_x[i]), list(trail_y[i]))
            markers[i].set_offsets(np.array([[x, y]]))
            labels[i].set_position((x + 0.05, y + 0.05))

        info.set_text(
            f"Day: {t_days:6.1f}\n"
            f"Barycenter-centered N-body simulation\n"
            f"Sun wobble is caused by the planets' gravity"
        )

        return trails + markers + labels + [info, footer]

    anim = FuncAnimation(fig, update, frames=frames, interval=33, blit=False)
    save_animation(anim, output_dir, "planetary_motion_nbody", fps=30, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    output_dir = find_desktop()

    for mode in MODES_TO_RENDER:
        print(f"Rendering: {mode}")
        if mode == "kepler_2d":
            render_kepler_2d(output_dir)
        elif mode == "kepler_3d":
            render_kepler_3d(output_dir)
        elif mode == "nbody":
            render_nbody(output_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    print(f"Done. Check your Desktop folder: {output_dir}")


if __name__ == "__main__":
    main()