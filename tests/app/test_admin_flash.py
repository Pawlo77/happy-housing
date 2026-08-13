"""Tests for Admin session flash banners."""

from unittest.mock import MagicMock, patch

from happy_housing.app.admin_flash import pop_admin_flash, set_admin_flash


def test_set_admin_flash_queues_multiple() -> None:
    session: dict[str, object] = {}
    fake_st = MagicMock()
    fake_st.session_state = session
    with patch("happy_housing.app.admin_flash.st", fake_st):
        set_admin_flash("success", "Updated 10 offers")
        set_admin_flash("info", "Next: recompute")
    assert session["admin_flash_messages"] == [
        ("success", "Updated 10 offers"),
        ("info", "Next: recompute"),
    ]


def test_pop_admin_flash_renders_and_clears() -> None:
    session: dict[str, object] = {
        "admin_flash_messages": [
            ("success", "Done"),
            ("info", "Tip"),
        ],
    }
    fake_st = MagicMock()
    fake_st.session_state = session
    with patch("happy_housing.app.admin_flash.st", fake_st):
        pop_admin_flash()
    fake_st.success.assert_called_once_with("Done")
    fake_st.info.assert_called_once_with("Tip")
    assert "admin_flash_messages" not in session


def test_pop_admin_flash_noop_when_empty() -> None:
    session: dict[str, object] = {}
    fake_st = MagicMock()
    fake_st.session_state = session
    with patch("happy_housing.app.admin_flash.st", fake_st):
        pop_admin_flash()
    fake_st.success.assert_not_called()
    fake_st.info.assert_not_called()
