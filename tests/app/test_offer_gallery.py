"""Tests for offer photo carousel index helpers."""

from happy_housing.app.offer_gallery import (
    clamp_photo_index,
    photo_fullscreen_key,
    photo_index_key,
    photo_nav_sync_key,
    photo_select_key,
    resolve_photo_select_label,
    should_keep_photo_dialog_open,
    wrap_photo_index,
)


def test_wrap_photo_index_wraps_around() -> None:
    assert wrap_photo_index(0, 5, -1) == 4
    assert wrap_photo_index(4, 5, 1) == 0
    assert wrap_photo_index(2, 5, 1) == 3


def test_wrap_photo_index_empty() -> None:
    assert wrap_photo_index(0, 0, 1) == 0


def test_clamp_photo_index() -> None:
    assert clamp_photo_index(-1, 3) == 0
    assert clamp_photo_index(99, 3) == 2
    assert clamp_photo_index(1, 3) == 1
    assert clamp_photo_index(0, 0) == 0


def test_photo_index_key() -> None:
    assert photo_index_key("OTO-1") == "offer_photo_idx_OTO-1"
    assert photo_select_key("OTO-1") == "offer_photo_sel_OTO-1"
    assert photo_nav_sync_key("OTO-1") == "offer_photo_nav_OTO-1"
    assert photo_fullscreen_key("OTO-1") == "offer_photo_fs_open_OTO-1"


def test_should_keep_photo_dialog_open_after_arrow_rerun() -> None:
    """Dialog must reopen on rerun after arrow nav (button no longer True)."""
    assert should_keep_photo_dialog_open(open_flag=True, open_clicked=False)
    assert should_keep_photo_dialog_open(open_flag=False, open_clicked=True)
    assert not should_keep_photo_dialog_open(open_flag=False, open_clicked=False)


def test_resolve_photo_select_label_syncs_after_arrow_nav() -> None:
    labels = ["Photo 1 / 3", "Photo 2 / 3", "Photo 3 / 3"]
    assert (
        resolve_photo_select_label(
            labels,
            idx=2,
            current_label="Photo 1 / 3",
            sync_from_index=True,
        )
        == "Photo 3 / 3"
    )


def test_resolve_photo_select_label_keeps_user_selectbox_choice() -> None:
    labels = ["Photo 1 / 3", "Photo 2 / 3", "Photo 3 / 3"]
    assert (
        resolve_photo_select_label(
            labels,
            idx=0,
            current_label="Photo 2 / 3",
            sync_from_index=False,
        )
        == "Photo 2 / 3"
    )


def test_resolve_photo_select_label_repairs_stale_label() -> None:
    labels = ["Photo 1 / 2", "Photo 2 / 2"]
    assert (
        resolve_photo_select_label(
            labels,
            idx=1,
            current_label="Photo 9 / 9",
            sync_from_index=False,
        )
        == "Photo 2 / 2"
    )
