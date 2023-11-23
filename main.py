import glob
import json
import os.path
from tqdm import tqdm
from src.tesseract import TesseractProcessor
from src.paddle import PaddleProcessor
from src.textract import TextractProcessor

if __name__ == '__main__':
    # processor = PaddleProcessor(ocr_version="PP-OCRv4", structure_version='PP-StructureV2')
    processor = TesseractProcessor()
    for path in tqdm(glob.glob("data/capture_screen/compression/*.jpeg")):
        name = os.path.basename(path)[:-5]
        output = processor.predict(path)
        json.dump(output, open(f"data/output/capture_screen/compression/{name}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
