import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ui_app

def test_ui_html_content():
    """Smoke test to ensure the UI module loads and HTML is valid."""
    assert "LunarIce-360" in ui_app.HTML
    assert "<button" in ui_app.HTML
