
import os
import re
import logging
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from pytesseract import Output


# ==========================================
# НАЛАШТУВАННЯ
# ==========================================

PDF_PATH = "Psalmy-Sions-ASD.pdf"
OUTPUT_PATH = "Psalmy-Sions-ASD.txt"

LANG = "ukr+eng+rus"
DPI = 300


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

POPPLER_PATH = r'C:\poppler-26.09.0\Library\bin'

# 3 — звичайний текстовий документ.
# 4 — переважно одна колонка.
# 6 — один суцільний текстовий блок.
PSM = 3

PREPROCESS = True
SAVE_DEBUG_IMAGES = False
MIN_CONFIDENCE = 0  # Не відкидаємо жодного тексту

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def preprocess_image(pil_image):
    """Покращує контраст і прибирає частину шуму."""

    image = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Локальне покращення контрасту
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Легке зменшення шуму
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=8, templateWindowSize=7,
        searchWindowSize=21
    )

    # Не застосовуємо жорстку бінаризацію:
    # вона іноді погіршує якість тонких літер.
    return Image.fromarray(gray)


def clean_text(text):
    """Прибирає зайві порожні рядки та пробіли."""

    text = text.replace("\x0c", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def recognize_page(image, lang=LANG, psm=PSM):
    """Розпізнає сторінку та повертає текст і впевненість."""

    config = (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=Output.DICT,
        timeout=180
    )

    words = []
    confidences = []

    for word, confidence in zip(
        data["text"], data["conf"]
    ):
        word = word.strip()

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            continue

        if not word or confidence < 0:
            continue

        words.append(word)
        confidences.append(confidence)

    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=config,
        timeout=180
    )

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    return clean_text(text), average_confidence


def scan_pdf_to_txt(
    pdf_path,
    output_txt_path,
    lang=LANG
):
    pdf_path = Path(pdf_path)
    output_path = Path(output_txt_path)

    if not pdf_path.is_file():
        logging.error("PDF-файл не знайдено: %s", pdf_path)
        return

    try:
        # Дізнаємося кількість сторінок без рендерингу всього PDF
        info = pdfinfo_from_path(
            str(pdf_path),
            poppler_path=POPPLER_PATH
        )
        total_pages = int(info["Pages"])

        logging.info("Сторінок у PDF: %d", total_pages)

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with output_path.open(
            "w", encoding="utf-8"
        ) as output_file:

            for page_number in range(1, total_pages + 1):
                logging.info(
                    "Обробка сторінки %d/%d",
                    page_number, total_pages
                )

                # Завантажуємо лише поточну сторінку
                images = convert_from_path(
                    str(pdf_path),
                    dpi=DPI,
                    first_page=page_number,
                    last_page=page_number,
                    grayscale=True,
                    poppler_path=POPPLER_PATH,
                    thread_count=1,
                    timeout=180
                )

                if not images:
                    logging.warning(
                        "Не вдалося отримати сторінку %d",
                        page_number
                    )
                    continue

                image = images[0]

                try:
                    if PREPROCESS:
                        processed = preprocess_image(image)
                    else:
                        processed = image

                    if SAVE_DEBUG_IMAGES:
                        debug_path = (
                            f"debug_page_{page_number}.png"
                        )
                        processed.save(debug_path)

                    text, confidence = recognize_page(
                        processed, lang=lang
                    )

                    output_file.write(
                        f"\n\n--- Сторінка {page_number} ---\n"
                    )
                    output_file.write(text + "\n")
                    output_file.flush()

                    logging.info(
                        "Сторінка %d: середня впевненість %.1f%%",
                        page_number, confidence
                    )

                    if confidence < 60:
                        logging.warning(
                            "Низька впевненість OCR. "
                            "Перевір сторінку вручну."
                        )

                finally:
                    image.close()

        logging.info(
            "Готово! Результат: %s",
            output_path.resolve()
        )

    except pytesseract.TesseractNotFoundError:
        logging.error(
            "Tesseract не знайдено. Перевір його встановлення "
            "та tesseract_cmd."
        )
    except pytesseract.TesseractError as error:
        logging.error("Помилка Tesseract: %s", error)
    except Exception as error:
        logging.exception("Помилка обробки PDF: %s", error)


if __name__ == "__main__":
    scan_pdf_to_txt(PDF_PATH, OUTPUT_PATH)
