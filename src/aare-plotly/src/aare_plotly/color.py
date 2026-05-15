# ruff: noqa: E741
from colorsys import rgb_to_hls, hls_to_rgb


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    # noinspection PyTypeChecker
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # pyright: ignore[reportReturnType]


def rgb_to_str(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def rgba_to_str(rgba: tuple[int, int, int, float]) -> str:
    return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"


def adjust_lightness(
    color: str,
    *,
    lightness_mult: float = 1.0,
    saturation_mult: float = 1.0,
    alpha: float | None = None,
) -> str:
    """
    Adjust lightness/saturation of a Plotly color.

    Supports:
      - '#RRGGBB'
      - 'rgb(r, g, b)'

    If ``alpha`` is given, the result is emitted as ``rgba(...)``.
    """
    if color.startswith("#"):
        r, g, b = hex_to_rgb(color)
    elif color.startswith("rgb"):
        r, g, b = map(int, color[4:-1].split(","))
    else:
        raise ValueError(f"Unsupported color format: {color}")

    h, l, s = rgb_to_hls(r / 255, g / 255, b / 255)

    l = max(0, min(1, l * lightness_mult))
    s = max(0, min(1, s * saturation_mult))

    r2, g2, b2 = hls_to_rgb(h, l, s)

    r_i, g_i, b_i = int(r2 * 255), int(g2 * 255), int(b2 * 255)

    if alpha is not None:
        return rgba_to_str((r_i, g_i, b_i, alpha))
    return rgb_to_str((r_i, g_i, b_i))
