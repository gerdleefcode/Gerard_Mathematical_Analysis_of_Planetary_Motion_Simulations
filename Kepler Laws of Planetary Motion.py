from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon


@dataclass(frozen=True)
class Planet:
    name: str
    a: float          # semi-major axis (AU)
    e: float          # eccentricity
    period: float     # orbital period (days)
    color: str
    phase: float = 0.0


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation: M = E - e sin(E)
    Returns eccentric anomaly E (radians).
    """
    M = float(M) % (2 * math.pi)
    E = M if e < 0.8 else math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        delta = -f / fp
        E += delta
        if abs(delta) < tol:
            break

    return E


def orbit_position(planet: Planet, t_days: float) -> tuple[float, float]:
    """
    Return x, y position in AU for a given time t_days.
    Sun is at the focus (0, 0).
    """
    M = 2.0 * math.pi * (t_days / planet.period) + planet.phase
    E = solve_kepler(M, planet.e)

    x = planet.a * (math.cos(E) - planet.e)
    y = planet.a * math.sqrt(1.0 - planet.e**2) * math.sin(E)
    return x, y


def orbit_curve(planet: Planet, n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Static orbit curve for drawing the ellipse."""
    E = np.linspace(0, 2 * np.pi, n)
    x = planet.a * (np.cos(E) - planet.e)
    y = planet.a * np.sqrt(1.0 - planet.e**2) * np.sin(E)
    return x, y


def find_desktop() -> Path:
    """
    Try common Desktop locations.
    Falls back to current working directory if none exist.
    """
    candidates = [
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",  # common on Windows
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path.cwd()


def main() -> None:
    plt.style.use("dark_background")

    # Approximate solar-system-like orbits
    planets = [
        Planet("Mercury", 0.39, 0.2056, 88.0,   "#b7b7b7", phase=0.0),
        Planet("Venus",   0.72, 0.0068, 224.7,  "#f7c948", phase=1.1),
        Planet("Earth",   1.00, 0.0167, 365.25, "#4cc9f0", phase=2.0),
        Planet("Mars",    1.52, 0.0934, 687.0,  "#ff6b6b", phase=2.7),
    ]

    days_per_frame = 2.0
    total_days = 730.0  # 2 years
    frames = int(total_days / days_per_frame) + 1

    limit = max(p.a * (1.0 + p.e) for p in planets) * 1.35

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#050816")
    ax.set_facecolor("#050816")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Planetary Motion with Kepler's Laws", fontsize=18, pad=16, color="white")

    # Star field
    rng = np.random.default_rng(7)
    sx = rng.uniform(-limit, limit, 300)
    sy = rng.uniform(-limit, limit, 300)
    ss = rng.uniform(2, 14, 300)
    ax.scatter(sx, sy, s=ss, c="white", alpha=0.15, linewidths=0, zorder=0)

    # Sun glow
    for size, alpha in [(2200, 0.06), (900, 0.14), (350, 0.95)]:
        ax.scatter([0], [0], s=size, c="#ffcf6b", alpha=alpha, edgecolors="none", zorder=10)

    trail_len = 120
    trail_lines = []
    markers = []
    labels = []
    trail_x = []
    trail_y = []

    # Draw orbits and initialize moving objects
    for planet in planets:
        ox, oy = orbit_curve(planet)
        ax.plot(ox, oy, color=planet.color, lw=1.2, alpha=0.35, zorder=1)

        trail_line, = ax.plot([], [], color=planet.color, lw=2.2, alpha=0.95, zorder=3)
        marker = ax.scatter([0], [0], s=75, c=planet.color, edgecolors="white",
                            linewidths=0.7, zorder=4)
        label = ax.text(0, 0, planet.name, fontsize=10, color="white",
                        ha="left", va="bottom", zorder=5)

        trail_lines.append(trail_line)
        markers.append(marker)
        labels.append(label)
        trail_x.append(deque(maxlen=trail_len))
        trail_y.append(deque(maxlen=trail_len))

    earth_idx = next(i for i, p in enumerate(planets) if p.name == "Earth")

    # Swept area triangle for Earth
    earth_wedge = Polygon(
        [[0, 0], [0, 0], [0, 0]],
        closed=True,
        facecolor=planets[earth_idx].color,
        edgecolor=planets[earth_idx].color,
        alpha=0.20,
        zorder=2,
    )
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
        fontsize=10, color="#d8e1ff",
        alpha=0.9,
        zorder=20,
    )

    def update(frame: int):
        t = frame * days_per_frame
        positions = [orbit_position(p, t) for p in planets]

        for i, (planet, (x, y)) in enumerate(zip(planets, positions)):
            trail_x[i].append(x)
            trail_y[i].append(y)

            trail_lines[i].set_data(trail_x[i], trail_y[i])
            markers[i].set_offsets(np.array([[x, y]]))
            labels[i].set_position((x + 0.05, y + 0.05))

        ex, ey = positions[earth_idx]
        if len(trail_x[earth_idx]) >= 2:
            px = trail_x[earth_idx][-2]
            py = trail_y[earth_idx][-2]
        else:
            px, py = ex, ey

        # Filled triangle showing the area swept since last frame
        earth_wedge.set_xy(np.array([[0, 0], [px, py], [ex, ey], [0, 0]]))
        swept_area = 0.5 * abs(px * ey - py * ex)

        info.set_text(
            f"Day: {t:6.1f}\n"
            f"Earth swept area over {days_per_frame:g} days: {swept_area:.5f} AU²\n"
            f"Areal speed ≈ {swept_area / days_per_frame:.5f} AU²/day"
        )

        return trail_lines + markers + labels + [earth_wedge, info, footer]

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=33,
        blit=False,
    )

    desktop = find_desktop()

    mp4_path = desktop / "kepler_planetary_motion.mp4"
    gif_path = desktop / "kepler_planetary_motion.gif"

    try:
        writer = FFMpegWriter(fps=30, bitrate=2200)
        anim.save(mp4_path, writer=writer, dpi=160)
        print(f"Saved animation to: {mp4_path}")
    except Exception as exc:
        print(f"MP4 save failed ({exc}). Falling back to GIF...")
        writer = PillowWriter(fps=20)
        anim.save(gif_path, writer=writer, dpi=120)
        print(f"Saved animation to: {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()