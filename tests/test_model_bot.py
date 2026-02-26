from tileflow.model_bot import _extract_description_and_images


def test_extract_description_and_images_from_markdown() -> None:
    readme = """
---
license: apache-2.0
---
# My Model

Fast model for chat and coding.

![preview](https://example.com/preview.png)
Some extra paragraph.
<img src="https://example.com/chart.jpg" />
"""
    description, images = _extract_description_and_images(readme)
    assert "Fast model for chat and coding." in description
    assert "https://example.com/preview.png" in images
    assert "https://example.com/chart.jpg" in images

