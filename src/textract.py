from typing import *
from PIL import Image
from textractor import Textractor


class TextractProcessor:

    def __init__(self) -> None:
        self.textractor = Textractor(region_name="us-east-1")

    def predict(self, path: str) -> List[Dict]:
        image = Image.open(path)
        width, height = image.size
        res = self.textractor.detect_document_text(path)
        output = []
        for word in res.words:
            output.append({
                "x0": int(width * word.x),
                "x1": int(width * (word.x + word.width)),
                "y0": int(height * word.y),
                "y1": int(height * (word.y + word.height)),
                "text": word.text,
                "type": "word"
            })

        for check in res.checkboxes:
            output.append({
                "x0": int(width * check.x),
                "x1": int(width * (check.x + check.width)),
                "y0": int(height * check.y),
                "y1": int(height * (check.y + check.height)),
                "text": "",
                "type": "checkbox"
            })
        return output
