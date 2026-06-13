from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np
import cv2

from app.detector import Detection
from app.batch_processor import BatchProcessor
from app.logger import CsvResultLogger
from app.mosaic import apply_mask_mosaic, apply_mosaic, build_mask, expand_segmentation_mask
from app.pdf_export import create_image_pdf, normalize_pdf_filename
from app.settings import (
    DEFAULT_SETTINGS,
    auto_pixel_size,
    load_settings,
    normalize_model_names,
    parse_class_ids,
    parse_class_names,
    resolve_target_class_ids,
    save_settings,
)
from app.utils import resolve_output_directory, unique_output_path
from app.utils import save_image


class SettingsTests(unittest.TestCase):
    def test_nsfw_detection_defaults(self) -> None:
        self.assertEqual(DEFAULT_SETTINGS["confidence"], 0.30)
        self.assertEqual(DEFAULT_SETTINGS["model_path"], "models/nsfw-anime-xl-x1280.pt")
        self.assertEqual(DEFAULT_SETTINGS["inference_size"], 1280)
        self.assertTrue(DEFAULT_SETTINGS["high_recall_detection"])
        self.assertEqual(DEFAULT_SETTINGS["target_class_mode"], "names")
        self.assertEqual(DEFAULT_SETTINGS["target_class_names"], ["vagina", "penis"])
        self.assertEqual(DEFAULT_SETTINGS["pixel_size_mode"], "auto_1_80")
        self.assertEqual(DEFAULT_SETTINGS["recommended_settings"]["confidence"], 0.30)

    def test_parse_class_ids(self) -> None:
        self.assertEqual(parse_class_ids("1, 0,1"), [0, 1])
        with self.assertRaises(ValueError):
            parse_class_ids("")
        with self.assertRaises(ValueError):
            parse_class_ids("-1")

    def test_auto_pixel_size(self) -> None:
        self.assertEqual(auto_pixel_size(2048, 1024, "auto_1_80", 16), 26)
        self.assertEqual(auto_pixel_size(800, 600, "fixed", 18), 18)

    def test_class_name_resolution(self) -> None:
        names = normalize_model_names({0: "nipple", 1: "vagina", 2: "penis", 3: "anus"})
        self.assertEqual(parse_class_names("Vagina, penis,vagina"), ["vagina", "penis"])
        self.assertEqual(resolve_target_class_ids(names, "names", [], ["vagina", "penis"]), [1, 2])
        self.assertEqual(resolve_target_class_ids(names, "ids", [2, 1], []), [1, 2])

    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            values = load_settings(path)
            values["confidence"] = 0.55
            save_settings(values, path)
            self.assertEqual(load_settings(path)["confidence"], 0.55)


class MosaicTests(unittest.TestCase):
    def test_bbox_mask_and_mosaic(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[:, :, 1] = np.arange(120, dtype=np.uint8)[None, :]
        detection = Detection((30, 25, 80, 75), 0, 0.9, "vagina")
        mask = build_mask((100, 120), [detection], 0.2, 4)
        self.assertGreater(float(mask[50, 55]), 0.9)
        self.assertEqual(float(mask[0, 0]), 0.0)
        output, pixel_size, _ = apply_mosaic(image, [detection], "fixed", 12, 0.2, 4)
        self.assertEqual(pixel_size, 12)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.array_equal(output[0, 0], image[0, 0]))

    def test_segmentation_mask_is_expanded_and_preferred(self) -> None:
        image = np.dstack([((np.indices((100, 120)).sum(axis=0) % 2) * 255).astype(np.uint8)] * 3)
        segmentation = np.zeros((100, 120), dtype=np.float32)
        cv2.circle(segmentation, (60, 50), 8, 1.0, -1)
        detection = Detection((45, 35, 75, 65), 1, 0.95, "vagina", segmentation)
        raw_area = int(np.count_nonzero(segmentation))
        expanded = expand_segmentation_mask(segmentation, 0.20)
        self.assertGreater(int(np.count_nonzero(expanded)), raw_area)
        mask = build_mask((100, 120), [detection], 0.20, 0)
        self.assertGreater(float(mask[50, 60]), 0.9)
        self.assertEqual(float(mask[10, 10]), 0.0)
        output, _, _ = apply_mosaic(image, [detection], "fixed", 10, 0.20, 4)
        self.assertFalse(np.array_equal(output[45:55, 55:65], image[45:55, 55:65]))

    def test_expand_ratio_applies_to_each_edge(self) -> None:
        from app.mosaic import _expanded_bbox

        self.assertEqual(_expanded_bbox((20, 20, 60, 60), (100, 100), 0.25), (10, 10, 70, 70))

    def test_manual_brush_mask_only_changes_painted_area(self) -> None:
        pattern = ((np.indices((80, 100)).sum(axis=0) % 2) * 255).astype(np.uint8)
        image = np.dstack([pattern, pattern, pattern])
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        output = apply_mask_mosaic(image, mask, 10)
        self.assertTrue(np.array_equal(output[0, 0], image[0, 0]))
        self.assertFalse(np.array_equal(output[30:50, 35:55], image[30:50, 35:55]))


class FileTests(unittest.TestCase):
    def test_unique_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "image.png"
            first.write_bytes(b"x")
            self.assertEqual(unique_output_path(folder, "image.png").name, "image_001.png")

    def test_empty_output_creates_numbered_folder_in_input(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            input_dir = Path(folder)
            first, first_created = resolve_output_directory(input_dir, "")
            second, second_created = resolve_output_directory(input_dir, "")
            self.assertTrue(first_created)
            self.assertTrue(second_created)
            self.assertEqual(first.name, "mosaic_output")
            self.assertEqual(second.name, "mosaic_output_001")

    def test_create_pdf_from_images(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = []
            for index, shape in enumerate(((60, 80), (80, 60)), start=1):
                path = root / f"{index:03d}.png"
                image = np.full((shape[0], shape[1], 3), index * 70, dtype=np.uint8)
                save_image(path, image)
                images.append(path)
            destination, page_count = create_image_pdf(images, root / "book.pdf")
            self.assertEqual(page_count, 2)
            self.assertTrue(destination.is_file())
            self.assertTrue(destination.read_bytes().startswith(b"%PDF"))
            self.assertEqual(normalize_pdf_filename("sample"), "sample.pdf")
            self.assertEqual(normalize_pdf_filename("bad:name.pdf"), "bad_name.pdf")


class BatchFilteringTests(unittest.TestCase):
    def test_only_selected_classes_are_masked_and_logged(self) -> None:
        class FakeDetector:
            def predict(
                self, image: np.ndarray, confidence: float, inference_size: int = 1280,
                high_recall: bool = False, tile_overlap: float = 0.20,
            ) -> list[Detection]:
                return [
                    Detection((10, 10, 45, 45), 1, 0.91, "vagina", np.pad(np.ones((20, 20), dtype=np.float32), ((10, 40), (10, 70)))),
                    Detection((55, 10, 90, 45), 0, 0.88, "nipple"),
                ]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.png"
            image = np.tile(np.arange(100, dtype=np.uint8), (70, 1))
            image = np.dstack([image, image, image])
            save_image(source, image)
            settings = {
                "confidence": 0.30,
                "pixel_size_mode": "fixed",
                "pixel_size": 12,
                "expand_ratio": 0.15,
                "feather_px": 8,
            }
            logger = CsvResultLogger(root / "result.csv")
            processor = BatchProcessor(settings)
            result = processor._process_one(FakeDetector(), [1], source, root / "out", root / "previews", 1, 1, logger)  # type: ignore[arg-type]
            self.assertEqual(result.detection_count, 2)
            self.assertEqual(result.masked_count, 1)
            self.assertEqual(result.status, "mosaic")
            with logger.path.open("r", newline="", encoding="utf-8-sig") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual(row["detected_classes"], "vagina;nipple")
            self.assertEqual(row["masked_classes"], "vagina")


class DetectionMergeTests(unittest.TestCase):
    def test_duplicate_boxes_keep_highest_confidence(self) -> None:
        from app.detector import YoloDetector

        detections = [
            Detection((10, 10, 50, 50), 2, 0.91, "penis"),
            Detection((12, 12, 51, 51), 2, 0.72, "penis"),
            Detection((12, 12, 51, 51), 1, 0.80, "vagina"),
        ]
        merged = YoloDetector._deduplicate(detections)
        self.assertEqual(len(merged), 2)
        self.assertEqual(max(item.confidence for item in merged if item.class_id == 2), 0.91)


if __name__ == "__main__":
    unittest.main()
