"""Tests for continuous Warsaw noise BitmapLayer overlay."""

from pathlib import Path

from PIL import Image

from happy_housing.app.maps import warsaw_noise_bitmap_layer
from happy_housing.gis.warsaw_noise import (
    NoiseMapOverlay,
    NoiseRaster,
    db_to_legend_rgba,
    prepare_noise_map_overlay,
    prepare_total_noise_map_overlay,
)


def _write_meta(root: Path, *, width: int = 2, height: int = 2) -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps({
            "bbox_wgs84": [20.9, 52.2, 21.1, 52.3],
            "bbox_2178": [0.0, 0.0, 2.0, 2.0],
            "width": width,
            "height": height,
            "crs": "EPSG:2178",
            "layers": {"road_lden": "HALAS_DROGOWY_LDWN_2022"},
        }),
        encoding="utf-8",
    )


def _road_raster(tmp_path: Path) -> dict[str, NoiseRaster]:
    root = tmp_path / "warsaw_noise"
    _write_meta(root)
    img = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    img.putpixel((0, 0), (252, 138, 30, 255))  # 55
    img.putpixel((1, 1), (177, 126, 217, 255))  # 70
    path = root / "road_lden.png"
    img.save(path)
    return {
        "road_lden": NoiseRaster(
            path=path,
            layer_key="road_lden",
            out_col="road_lden_db",
            xmin=0.0,
            ymin=0.0,
            xmax=2.0,
            ymax=2.0,
            width=2,
            height=2,
        )
    }


def test_db_to_legend_rgba_quiet_is_transparent() -> None:
    assert db_to_legend_rgba(0.0)[3] == 0
    assert db_to_legend_rgba(40.0)[3] == 0
    assert db_to_legend_rgba(55.0)[:3] == (252, 138, 30)
    assert db_to_legend_rgba(55.0)[3] > 0


def test_prepare_noise_map_overlay_single_layer_is_continuous(tmp_path: Path) -> None:
    rasters = _road_raster(tmp_path)
    overlay = prepare_noise_map_overlay(
        rasters,
        layer_key="road_lden",
        bbox_wgs84=(20.9, 52.2, 21.1, 52.3),
    )
    assert overlay is not None
    assert overlay.image_data_uri.startswith("data:image/png;base64,")
    assert overlay.bounds == (20.9, 52.2, 21.1, 52.3)
    # Full raster, not a thinned point cloud.
    assert overlay.width == 2
    assert overlay.height == 2


def test_prepare_total_noise_map_overlay_composites_sources(tmp_path: Path) -> None:
    rasters = _road_raster(tmp_path)
    # Duplicate as rail so energy sum has two sources on one pixel.
    rail_path = tmp_path / "warsaw_noise" / "rail_lden.png"
    Image.open(rasters["road_lden"].path).save(rail_path)
    rasters["rail_lden"] = NoiseRaster(
        path=rail_path,
        layer_key="rail_lden",
        out_col="rail_lden_db",
        xmin=0.0,
        ymin=0.0,
        xmax=2.0,
        ymax=2.0,
        width=2,
        height=2,
    )
    overlay = prepare_total_noise_map_overlay(
        rasters,
        indicator="lden",
        bbox_wgs84=(20.9, 52.2, 21.1, 52.3),
    )
    assert overlay is not None
    assert "total_lden" in overlay.layer_key
    assert overlay.image_data_uri.startswith("data:image/png;base64,")


def test_warsaw_noise_bitmap_layer_uses_bitmap() -> None:
    overlay = NoiseMapOverlay(
        layer_key="road_lden",
        image_data_uri="data:image/png;base64,aaa",
        bounds=(20.9, 52.2, 21.1, 52.3),
        width=2,
        height=2,
    )
    layer = warsaw_noise_bitmap_layer(overlay)
    assert layer is not None
    payload = layer.to_json()
    assert "BitmapLayer" in payload
    assert "HeatmapLayer" not in payload
    assert list(layer.bounds) == [20.9, 52.2, 21.1, 52.3]
