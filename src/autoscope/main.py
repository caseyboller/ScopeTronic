"""AutoScope entry point.

Basic CLI skeleton. Extend the pipeline steps in the ``Autoscope`` class.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("autoscope")


@dataclass
class Config:
    """Runtime configuration for the pipeline."""

    input_path: Path
    output_path: Path
    verbose: bool = False


class Autoscope:
    """Main processing pipeline.

    Each stage is intentionally a small, testable method. Replace the
    placeholder bodies with the real implementation.
    """

    scale_factor = 0.5

    def new_tracker(self):
        #return cv2.legacy.TrackerCSRT_create()
        return cv2.legacy.TrackerKCF_create()

    def loop(self):
        cap = cv2.VideoCapture(".\\sample_videos\\video.mp4") # Use 0 for camera, or "sample.mp4" for file
        tracker = self.new_tracker()
        tracking_active = False
        state = "IDLE"
        last_bbox = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or camera disconnected.")
                break

            frame = self.convert_image_resize(frame)
            frame = self.convert_image_grayscale(frame)

            if state == "TRACKING":
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    self.compute_frame_offsets(frame, bbox)
                    last_bbox = bbox
                else:
                    cv2.putText(frame, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    state = "SEARCHING"

            elif state == "SEARCHING":
                recovered_box = self.find_specimen_fallback(frame, last_bbox)
                if recovered_box:
                    print("Specimen re-acquired!")
                    tracker = cv2.legacy.TrackerKCF_create() # Re-initialize tracker
                    tracker.init(frame, recovered_box)
                    state = "TRACKING"
                else:
                    # If search fails for too many frames, pause and ask user or stop stage
                    cv2.putText(frame, "Uh oh!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    state = "SEARCHING"
            cv2.imshow("Auto Scope Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Pause stream and select region of interest (ROI)
                bbox = cv2.selectROI("Auto Scope Live Feed", frame, fromCenter=False, showCrosshair=True)
                # bbox returns (x, y, w, h)
                print(f"Selected bounding box: {bbox}")
                if bbox[2] > 0 and bbox[3] > 0:
                    state = "TRACKING"
                    tracker.init(frame, bbox)

            elif key == ord('c'):
                state = "IDLE"
                tracker = self.new_tracker()

            # Press 'q' to exit
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    def convert_image_grayscale(self, frame):
        # Inside your loop, right after reading the frame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        # Convert back to 3-channel so your display and tracker still get a standard image
        processed_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return processed_frame

    def convert_image_resize(self, frame):
        width = int(frame.shape[1] * self.scale_factor)
        height = int(frame.shape[0] * self.scale_factor)
        small_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return small_frame

    def compute_frame_offsets(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        height, width, _ = frame.shape
        frame_center_x, frame_center_y = width // 2, height // 2

        obj_center_x = x + (w // 2)
        obj_center_y = y + (h // 2)

        dx = obj_center_x - frame_center_x
        dy = obj_center_y - frame_center_y

        dx *= 1/self.scale_factor
        dy *= 1/self.scale_factor

        # Display crosshairs and offset text
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), (255, 0, 0), 1)
        cv2.line(frame, (0, frame_center_y), (width, frame_center_y), (255, 0, 0), 1)
        cv2.putText(frame, f"Offset: dx={dx}, dy={dy}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def find_specimen_fallback(self, frame, last_known_box):
        """
        Attempts to re-detect the object using simple thresholding and contours 
        near the last known location.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding (great for uneven microscope lighting)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter contours by a reasonable size (avoid tiny specks or huge background shifts)
        lx, ly, lw, lh = last_known_box
        best_box = None
        min_distance = float('inf')
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out obvious noise based on expected object size (e.g., width/height limits)
            if 10 < w < 200 and 10 < h < 200:
                # Find the contour closest to the last known center
                cx, cy = x + w//2, y + h//2
                last_cx, last_cy = lx + lw//2, ly + lh//2
                distance = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                
                if distance < min_distance and distance < 150: # Must be within 150 pixels of last location
                    min_distance = distance
                    best_box = (x, y, w, h)
                    
        return best_box


    ####################

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> int:
        """Execute the full pipeline. Returns a process exit code."""
        logger.info("AutoScope starting")
        logger.debug("Config: %s", self.config)

        try:
            frame = self.load()
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1

        result = self.process(frame)
        self.save(result)

        logger.info("AutoScope finished")
        return 0

    def load(self) -> np.ndarray:
        """Load the input image from disk."""
        path = self.config.input_path
        logger.info("Loading input: %s", path)

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply the processing pipeline to a single frame.

        TODO: insert real processing steps here (filters, detection, etc.).
        """
        logger.info("Processing frame with shape %s", frame.shape)
        # Placeholder: return the frame unchanged.
        return frame

    def save(self, frame: np.ndarray) -> None:
        """Write the result to disk."""
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving output: %s", path)
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write image: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="autoscope",
        description="AutoScope - automatic image processing pipeline.",
    )
    parser.add_argument("input", type=Path, help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output.png"),
        help="Path to write the processed image (default: output.png).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    # args = build_parser().parse_args(argv)

    # logging.basicConfig(
    #     level=logging.DEBUG if args.verbose else logging.INFO,
    #     format="%(levelname)s: %(message)s",
    # )

    # config = Config(
    #     input_path=args.input,
    #     output_path=args.output,
    #     verbose=args.verbose,
    # )
    config = []
    return Autoscope(config).loop()


if __name__ == "__main__":
    sys.exit(main())
