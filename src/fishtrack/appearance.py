"""Confidence-gated appearance following, independent of motion detections."""

import cv2
import numpy as np
from numpy.typing import NDArray

from fishtrack.detection import Detection, Image


def crop_patch(gray: Image, detection: Detection) -> Image | None:
    """Copy an in-frame, textured bounding-box patch or return None.

    :param gray: Grayscale frame.
    :param detection: Region in full-resolution pixel coordinates.
    :return: Independent grayscale appearance patch suitable for correlation.
    """

    # Reject unusable crops before building an appearance model.
    x, y, width, height = detection.box
    if width < 8 or height < 8 or x < 0 or y < 0:
        return None
    if x + width > gray.shape[1] or y + height > gray.shape[0]:
        return None

    # Flat regions provide too little visual detail for reliable tracking.
    patch = gray[y : y + height, x : x + width]
    if float(np.std(patch)) < 5:
        return None
    return patch.copy()


def optical_prediction(
    previous: Image, current: Image, detection: Detection
) -> tuple[float, float] | None:
    """Estimate object translation with forward/backward checked local LK features.

    :param previous: Previous full-frame grayscale reference.
    :param current: Current grayscale frame of equal dimensions.
    :param detection: Object contour in previous-frame coordinates.
    :return: Predicted centroid, or None when too few consistent features survive.
    """

    # Bound optical-flow cost while keeping the final prediction in source pixels.
    scale = min(1.0, 960 / previous.shape[1])
    size = (round(previous.shape[1] * scale), round(previous.shape[0] * scale))
    old = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    new = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    # Look for features inside the last contour rather than on the surrounding reef.
    mask = np.zeros_like(old)
    contour = np.rint(detection.contour * scale).astype(np.int32)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(old, 50, 0.01, 3, mask=mask)
    if points is None or len(points) < 4:
        return None
    following, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
    if following is None or status is None:
        return None

    # A feature must survive a round trip to reject ambiguous correspondences.
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(new, old, following, None)
    if backward is None or backward_status is None:
        return None
    valid = (status.ravel() == 1) & (backward_status.ravel() == 1)
    valid &= np.isfinite(following).all(axis=(1, 2))
    valid &= np.linalg.norm((backward - points).reshape(-1, 2), axis=1) < 1.5
    if np.count_nonzero(valid) < 4:
        return None

    # Require a shared displacement so a few background features cannot steer the track.
    displacements = (following[valid] - points[valid]).reshape(-1, 2)
    shift = np.median(displacements, axis=0)
    consistent = np.linalg.norm(displacements - shift, axis=1) < 2.0
    if np.count_nonzero(consistent) < 4 or np.mean(consistent) < 0.6:
        return None
    shift = np.median(displacements[consistent], axis=0) / scale
    return detection.centroid[0] + float(shift[0]), detection.centroid[1] + float(shift[1])


def appearance_histogram(frame: Image, box: tuple[int, int, int, int]) -> NDArray[np.float32]:
    """Describe central object colour without requiring a fixed pose.

    :param frame: BGR image.
    :param box: In-frame object rectangle.
    :return: Normalized HSV histogram; an ellipse reduces surrounding-background influence.
    """
    x, y, width, height = box
    patch = cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2HSV)

    # Weight the box centre to reduce the influence of background at its corners.
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(1, width // 2 - 1), max(1, height // 2 - 1)),
        0,
        0,
        360,
        255,
        -1,
    )
    histogram = cv2.calcHist([patch], [0, 1, 2], mask, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(histogram, None, alpha=1, norm_type=cv2.NORM_L1)


class AppearanceFollower:
    """Adaptive CSRT appearance tracking checked against local optical-flow prediction."""

    def __init__(self, frame: Image, detection: Detection, threshold: float) -> None:
        """Initialise a visual model from one accepted region.

        :param frame: BGR frame containing the region.
        :param detection: Object coordinates in this frame.
        :param threshold: CSRT response threshold; higher values reject weaker matches.
        """
        self.failure_reason: str | None = None
        self.initial_box = detection.box
        self.histogram = appearance_histogram(frame, detection.box)

        # Run CSRT at a bounded resolution, retaining source coordinates for other checks.
        self.scale = min(1.0, 960 / frame.shape[1])
        self.size = (round(frame.shape[1] * self.scale), round(frame.shape[0] * self.scale))
        parameters = cv2.TrackerCSRT_Params()
        parameters.psr_threshold = threshold
        self.tracker = cv2.TrackerCSRT_create(parameters)
        box = tuple(round(value * self.scale) for value in detection.box)
        if min(box[2:]) < 4:
            raise ValueError("Selected region is too small at tracking resolution")
        self.tracker.init(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), box)

    def update(
        self,
        frame: Image,
        gray: Image,
        previous: Image | None,
        detection: Detection,
        expected: tuple[float, float],
        max_distance: float,
    ) -> Detection | None:
        """Update appearance and return a supported region, never an unobserved prediction.

        :param frame: Current BGR frame.
        :param gray: Current grayscale frame.
        :param previous: Previous grayscale frame when the object was visible there.
        :param detection: Last accepted contour and box.
        :param expected: Camera-based centroid prediction.
        :param max_distance: Maximum deviation from optical/camera prediction in pixels.
        :return: Scaled/translated contour on success, otherwise None.
        """
        self.failure_reason = None
        success, box = self.tracker.update(
            cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        )
        if not success:
            self.failure_reason = "weak_response"
            return None
        x, y, width, height = (round(value / self.scale) for value in box)
        if x < 0 or y < 0 or width < 8 or height < 8:
            self.failure_reason = "out_of_frame"
            return None
        if x + width > frame.shape[1] or y + height > frame.shape[0]:
            self.failure_reason = "out_of_frame"
            return None

        # Reject gradual box expansion or shrinkage relative to the original selection.
        if not (
            0.6 <= width / self.initial_box[2] <= 1.8 and 0.6 <= height / self.initial_box[3] <= 1.8
        ):
            self.failure_reason = "size_drift"
            return None

        # Compare against the original colour model to catch drift onto nearby scenery.
        histogram = appearance_histogram(frame, (x, y, width, height))
        if cv2.compareHist(self.histogram, histogram, cv2.HISTCMP_BHATTACHARYYA) > 0.65:
            self.failure_reason = "colour_change"
            return None
        old_x, old_y, old_width, old_height = detection.box
        if not (0.5 <= width / old_width <= 2 and 0.5 <= height / old_height <= 2):
            self.failure_reason = "size_jump"
            return None

        # Prefer local object motion when it is reliable; otherwise use camera prediction.
        if previous is not None:
            optical = optical_prediction(previous, gray, detection)
            if optical is not None:
                expected = optical
        centroid = (x + (width - 1) / 2, y + (height - 1) / 2)
        if np.linalg.norm(np.subtract(centroid, expected)) > max_distance:
            self.failure_reason = "motion_disagreement"
            return None

        # Transport the last contour; appearance tracking does not segment a new silhouette.
        contour = detection.contour.astype(np.float64) - (old_x, old_y)
        contour *= ((width - 1) / max(1, old_width - 1), (height - 1) / max(1, old_height - 1))
        contour += (x, y)
        return Detection(np.rint(contour).astype(np.int32), centroid, (x, y, width, height))
