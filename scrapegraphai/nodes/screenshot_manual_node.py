"""
fetch_screen_node module
"""

import asyncio
from typing import List, Optional
from .base_node import BaseNode
from ..utils.screenshot_scraping import (
    take_screenshot,
    select_area_with_opencv,
    crop_image,
    detect_text
)

class ScreenShotManualNode(BaseNode):
    """
    ScreenShotManualNode captures screenshots from a given URL
    and stores the image data as bytes.
    The node uses OCR to recognize text on the captured image and saves 
    it in an output field named "text".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "ScreenShotManualNode",
    ):
        """
        Initializes an instance of ScreenShotManualNode.

        Args:
            input (str): The name of the input field.
            output (List[str]): The names of the output fields.
            node_config (Optional[dict]): Additional configuration for the node.
            node_name (str): The name of the node.
        """
        super().__init__(node_name, "node", input, output, 2, node_config)
        self.url = node_config.get("link")

    async def execute(self, state: dict) -> dict:
        """
        Executes the node to capture a screenshot and recognize text.

        Args:
            state (dict): The current state of the node.

        Returns:
            dict: The updated state of the node with the recognized text.
        """
        image = await take_screenshot(
            url=self.url,
            save_path="Savedscreenshots/test_image.jpeg",
            quality=50
        )

        LEFT, TOP, RIGHT, BOTTOM = select_area_with_opencv(image)
        print("LEFT: ", LEFT, " TOP: ", TOP, " RIGHT: ", RIGHT, " BOTTOM: ", BOTTOM)

        cropped_image = crop_image(image, LEFT=LEFT, RIGHT=RIGHT, TOP=TOP, BOTTOM=BOTTOM)

        text = detect_text(
            cropped_image,
            languages=["en"]
        )

        state.update({self.output[0]: text})
        return state
