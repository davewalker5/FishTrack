"""Estimate camera motion and extract approximate independent-motion contours."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass
class Detection:
    """An approximate object contour in current-frame pixel coordinates."""

    contour: NDArray[np.int32]
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionConfig:
    """Motion intensity, pixel-area and shape thresholds."""

    motion_threshold: float = 25
    min_area: float = 80
    max_area: float = 50000
    max_aspect_ratio: float = 8
    compensate: bool = True
    local_motion: bool = True
    min_residual_motion: float = 2.0
    suppress_unreliable: bool = True

    def __post_init__(self) -> None:
        """Reject nonfinite or inconsistent thresholds."""

        # Validate thresholds up front so invalid values cannot silently disable filtering.
        values = (self.motion_threshold, self.min_area, self.max_area, self.max_aspect_ratio)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Detection thresholds must be finite")
        if not np.isfinite(self.min_residual_motion) or self.min_residual_motion <= 0:
            raise ValueError("min_residual_motion must be finite and positive")
        if not 0 < self.motion_threshold <= 255:
            raise ValueError("motion_threshold must be in (0, 255]")
        if not 0 < self.min_area <= self.max_area or self.max_aspect_ratio < 1:
            raise ValueError("Require 0 < min_area <= max_area and max_aspect_ratio >= 1")


def estimate_motion(previous: Image, current: Image) -> tuple[NDArray[np.float64], bool]:
    """Estimate previous-to-current affine motion using LK features and RANSAC.

    :param previous: Previous grayscale frame.
    :param current: Current grayscale frame of the same dimensions.
    :return: Pixel-coordinate affine matrix and whether a reliable fit was found.
    """

    # Return an explicit unreliable identity transform when camera motion cannot be fitted.
    identity = np.eye(2, 3, dtype=np.float64)
    points = cv2.goodFeaturesToTrack(previous, 500, 0.01, 10)
    if points is None or len(points) < 8:
        return identity, False
    following, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if following is None or status is None:
        return identity, False
    valid = status.ravel() == 1
    valid &= np.isfinite(following).all(axis=(1, 2))
    if np.count_nonzero(valid) < 8:
        return identity, False

    # RANSAC reduces the influence of independently moving fish on the camera estimate.
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[valid], following[valid], method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if matrix is None or inliers is None or np.mean(inliers) < 0.5:
        return identity, False

    # Reject implausible zoom estimates even if the feature fit produced a matrix.
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    if not np.isfinite(matrix).all() or not 0.8 <= scale <= 1.25:
        return identity, False
    return matrix, True


def extract_detections(mask: Image, config: DetectionConfig) -> list[Detection]:
    """Filter external mask contours by area and bounding-box aspect ratio.

    :param mask: Binary single-channel candidate mask.
    :param config: Area limits in pixels squared and dimensionless aspect limit.
    :return: Valid contours with centroids and bounding boxes.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    # Remove implausible sizes and shapes before spending time on local motion checks.
    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.min_area <= area <= config.max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if max(width / height, height / width) > config.max_aspect_ratio:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        detections.append(Detection(contour, centroid, (x, y, width, height)))
    return detections


def filter_local_motion(
    previous: Image, current: Image, detections: list[Detection], minimum: float
) -> list[Detection]:
    """Keep contours whose coherent flow differs from their surrounding background.

    :param previous: Previous grayscale frame.
    :param current: Current grayscale frame.
    :param detections: Current-frame candidate contours.
    :param minimum: Minimum median residual speed in original pixels per frame.
    :return: Candidates with locally independent motion; ambiguous regions are rejected.
    """
    if not detections:
        return []

    # Dense flow is expensive, so calculate it on a smaller image and rescale speeds.
    scale = min(1.0, 640 / current.shape[1])
    size = (round(current.shape[1] * scale), round(current.shape[0] * scale))
    old = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    new = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    # Backward flow is located in current-frame coordinates, matching the contours.
    flow = cv2.calcOpticalFlowFarneback(new, old, None, 0.5, 4, 21, 3, 5, 1.2, 0)
    accepted = []
    for detection in detections:
        contour = np.rint(detection.contour * scale).astype(np.int32)
        x, y, width, height = cv2.boundingRect(contour)
        padding = 12
        left, top = max(0, x - padding), max(0, y - padding)
        right, bottom = min(size[0], x + width + padding), min(size[1], y + height + padding)
        region = np.zeros((bottom - top, right - left), np.uint8)
        cv2.drawContours(region, [contour - (left, top)], -1, 255, -1)

        # Compare the contour interior with a surrounding ring separated from its boundary.
        inner = cv2.erode(region, np.ones((3, 3), np.uint8)) > 0
        outer = cv2.dilate(region, np.ones((19, 19), np.uint8)) > 0
        excluded = cv2.dilate(region, np.ones((7, 7), np.uint8)) > 0
        ring = outer & ~excluded
        if np.count_nonzero(inner) < 4 or np.count_nonzero(ring) < 20:
            continue
        local = flow[top:bottom, left:right]

        # Subtract nearby background motion to distinguish fish from camera movement.
        background = np.median(local[ring], axis=0)
        residual = local[inner] - background

        # Median vector requires coherent motion rather than scattered noisy flow.
        speed = np.linalg.norm(np.median(residual, axis=0)) / scale
        spread = np.median(np.linalg.norm(local[ring] - background, axis=1)) / scale
        if speed >= max(minimum, 2 * spread):
            accepted.append(detection)
    return accepted


class MotionDetector:
    """Compare consecutive aligned frames without mutating input images."""

    def __init__(self, config: DetectionConfig) -> None:
        """Initialise detector state with validated configuration."""
        self.config = config

        # Keep only the preceding grayscale frame for frame-to-frame comparison.
        self.previous: Image | None = None
        self.suppression_reason: str | None = None

    def update(
        self, frame: Image, detect: bool = True
    ) -> tuple[list[Detection], Image, Image, NDArray, bool]:
        """Return detections, mask, aligned grayscale background, motion and fit status.

        :param frame: Three-channel BGR video frame.
        :param detect: Extract motion candidates; disable for manual-only tracking.
        :return: Results in current-frame coordinates; first frame has no detections.
        """
        self.suppression_reason = None

        # Reduce fine image noise before comparing brightness between frames.
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        matrix = np.eye(2, 3, dtype=np.float64)
        mask = np.zeros_like(gray)
        aligned = gray.copy()
        reliable = False
        if self.previous is not None:
            if self.config.compensate:
                matrix, reliable = estimate_motion(self.previous, gray)
            height, width = gray.shape
            aligned = cv2.warpAffine(self.previous, matrix, (width, height))

            # Exclude warped image borders, which have no corresponding source pixels.
            valid = cv2.warpAffine(np.full_like(gray, 255), matrix, (width, height))
            valid = cv2.erode(
                valid, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            difference = cv2.absdiff(gray, aligned)
            _, mask = cv2.threshold(
                difference, self.config.motion_threshold, 255, cv2.THRESH_BINARY
            )

            # Remove isolated noise and reconnect nearby parts of moving contours.
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.bitwise_and(mask, valid)
        detections = extract_detections(mask, self.config) if detect else []
        if self.previous is not None:

            # Suppress acquisition when scene changes or poor alignment make motion ambiguous.
            changed_fraction = np.mean(cv2.absdiff(gray, aligned) > 60)
            if self.config.suppress_unreliable and changed_fraction > 0.65:
                self.suppression_reason = "scene_change"
            elif self.config.suppress_unreliable and self.config.compensate and not reliable:
                self.suppression_reason = "unreliable_alignment"
            if self.suppression_reason is not None:
                detections = []
                mask[:] = 0
            elif self.config.local_motion:
                detections = filter_local_motion(
                    self.previous, gray, detections, self.config.min_residual_motion
                )

                # Diagnostics show only candidates surviving the local motion gate.
                filtered = np.zeros_like(mask)
                cv2.drawContours(filtered, [d.contour for d in detections], -1, 255, -1)
                mask = cv2.bitwise_and(mask, filtered)

        # Advance the reference even when detections were suppressed.
        self.previous = gray
        return detections, mask, aligned, matrix, reliable
