import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import Dict, List
from paddleocr import PaddleOCR


class PaddleProcessor:

    def __init__(self, ocr_version="PP-OCR", structure_version='PP-Structure') -> None:
        self.ocr = PaddleOCR(lang="en", ocr_version=ocr_version, structure_version=structure_version, use_gpu=True)

    def convert_res_to_output(self, res: Dict) -> List[Dict]:
        """Convert the tesseract_raw result to correct output format.

        Args:
            res (Dict): Raw result dict: dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text']).

        Returns:
            List[Dict]: List of converted object dicts.
        """

        output: List[Dict] = []
        if res:
            for box, (text, conf) in res:
                x0 = min(b[0] for b in box)
                x1 = max(b[0] for b in box)
                y0 = min(b[1] for b in box)
                y1 = max(b[1] for b in box)
                output.append({
                    'x0': x0,
                    'x1': x1,
                    'y0': y0,
                    'y1': y1,
                    'text': text,
                    "type": "word"
                })
        return output

    def predict(self, image) -> List[Dict]:
        res = self.ocr.ocr(image, cls=False)[0]
        return self.convert_res_to_output(res)
