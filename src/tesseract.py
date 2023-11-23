import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import Dict, List
import pytesseract
from pytesseract import Output

TESSERACT_CONFIDENCE_THRESHOLD = 0.3


class TesseractProcessor:

    def __init__(self):
        pass

    def convert_res_to_output(self, res: Dict) -> List[Dict]:
        """Convert the tesseract_raw result to correct output format.

        Args:
            res (Dict): Raw result dict: dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text']).

        Returns:
            List[Dict]: List of converted object dicts.
        """

        output: List[Dict] = []
        n_boxes = len(res['text'])
        for i in range(n_boxes):
            if int(res['conf'][i]) > TESSERACT_CONFIDENCE_THRESHOLD or len(res["text"][i]) >= 2:
                (x, y, w, h) = (res['left'][i], res['top'][i], res['width'][i], res['height'][i])
                output.append({
                    'x0': x,
                    'x1': x + w,
                    'y0': y,
                    'y1': y + h,
                    'text': res['text'][i],
                    "type": "word"
                })
        return output

    def predict(self, image) -> List[Dict]:
        res = pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT)
        return self.convert_res_to_output(res)
