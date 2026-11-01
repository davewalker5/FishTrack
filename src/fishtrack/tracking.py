"""Motion-based acquisition followed by independent, confidence-gated visual tracking."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fishtrack.appearance import AppearanceFollower, crop_patch
from fishtrack.detection import Detection, Image

LOGGER = logging.getLogger(__name__)


@dataclass
class Track:
    """Temporary identity, observation counts and independent appearance state."""

    identifier: int
    detection: Detection
    position: tuple[float, float]
    hits: int = 1
    missing: int = 0
    consecutive_hits: int = 1
    confirmed: bool = False
    propagated: bool = False
    follower: AppearanceFollower | None = None
    followed_frames: int = 0
    manual: bool = False

    def is_visible(self, min_hits: int) -> bool:
        """Return whether current visual evidence supports drawing this identity.

        :param min_hits: Consecutive detections needed before automatic acquisition.
        :return: False for unobserved predictions or unconfirmed detector-only regions.
        """

        # A retained identity is drawable only when this frame has supporting visual evidence.
        return self.missing == 0 and (
            self.consecutive_hits >= min_hits or (self.confirmed and self.follower is not None)
        )


def box_detection(box: tuple[int, int, int, int]) -> Detection:
    """Represent a selected rectangle as a contour, without claiming segmentation.

    :param box: Left, top, width and height in source-frame pixels.
    :return: A rectangular seed contour and centroid.
    :raises ValueError: If the rectangle has invalid dimensions or coordinates.
    """
    x, y, width, height = box
    if x < 0 or y < 0 or width < 8 or height < 8:
        raise ValueError("ROI requires nonnegative x/y and width/height of at least 8 pixels")

    # Represent a seed as a rectangle until a tracking method supplies a new observation.
    contour = np.array(
        [[[x, y]], [[x + width - 1, y]], [[x + width - 1, y + height - 1]], [[x, y + height - 1]]],
        dtype=np.int32,
    )
    return Detection(contour, (x + (width - 1) / 2, y + (height - 1) / 2), box)


class Tracker:
    """Acquire objects from detections and follow confirmed objects through detector gaps."""

    def __init__(
        self,
        max_distance: float = 60,
        max_missing: int = 5,
        min_hits: int = 3,
        appearance: bool = True,
        tracking_threshold: float = 0.035,
    ) -> None:
        """Configure association distance, loss tolerance and visual-following confidence.

        :param max_distance: Search and association gate in source-frame pixels.
        :param max_missing: Maximum consecutive unobserved frames before expiry.
        :param min_hits: Consecutive detections needed for automatic confirmation.
        :param appearance: Enable optical-flow prediction and appearance following.
        :param tracking_threshold: CSRT response threshold in (0, 1]; higher values are stricter.
        """
        if not np.isfinite(max_distance) or max_distance <= 0 or max_missing < 0:
            raise ValueError("Require finite positive max_distance and nonnegative max_missing")
        if min_hits < 1 or not np.isfinite(tracking_threshold) or not 0 < tracking_threshold <= 1:
            raise ValueError("Require positive min_hits and tracking_threshold in (0, 1]")
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.appearance = appearance
        self.tracking_threshold = tracking_threshold

        # Maintain identities across frames and issue IDs monotonically within a run.
        self.tracks: list[Track] = []
        self.next_identifier = 1
        self.previous_gray: Image | None = None

    def seed(self, frame: Image, boxes: tuple[tuple[int, int, int, int], ...]) -> None:
        """Create confirmed tracks for user-selected boxes in the current frame.

        :param frame: BGR frame on which the selections were made.
        :param boxes: One or more full-resolution x/y/width/height rectangles.
        :raises ValueError: If a region is outside the image or lacks usable texture.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Build every seed before committing, so one invalid selection cannot partially add tracks.
        seeds = []
        for box in boxes:
            detection = box_detection(box)
            anchor = crop_patch(gray, detection)
            if anchor is None:
                raise ValueError(f"ROI {box} is outside the frame or lacks trackable texture")
            seeds.append(
                Track(
                    self.next_identifier + len(seeds),
                    detection,
                    detection.centroid,
                    confirmed=True,
                    follower=AppearanceFollower(frame, detection, self.tracking_threshold),
                    manual=True,
                )
            )

        # Commit validated selections together and establish the optical-flow reference.
        self.next_identifier += len(seeds)
        self.tracks.extend(seeds)
        self.previous_gray = gray

    def reset(self) -> None:
        """Discard scene-specific state while preserving ID uniqueness within this run."""

        # Drop scene-specific models without recycling IDs already used in the output.
        self.tracks.clear()
        self.previous_gray = None

    def _new_follower(
        self, frame: Image, gray: Image, detection: Detection
    ) -> AppearanceFollower | None:
        """Create a model only for sufficiently large and textured automatic candidates."""

        # Reject candidates that become too small or featureless at the follower's resolution.
        scale = min(1.0, 960 / frame.shape[1])
        if min(round(value * scale) for value in detection.box[2:]) < 4:
            return None
        if crop_patch(gray, detection) is None:
            return None
        return AppearanceFollower(frame, detection, self.tracking_threshold)

    def _follow(self, track: Track, frame: Image, gray: Image, was_missing: int) -> bool:
        """Advance a confirmed track only with independent visual support."""
        if not self.appearance or not track.confirmed or track.follower is None:
            return False

        # Use local flow only when the preceding frame actually observed this fish.
        match = track.follower.update(
            frame,
            gray,
            self.previous_gray if was_missing == 0 else None,
            track.detection,
            track.position,
            self.max_distance,
        )
        if match is None:
            LOGGER.debug("Track %d not drawn: %s", track.identifier, track.follower.failure_reason)
            return False

        # Only accepted visual matches reset the missing counter and update the visible contour.
        track.detection = match
        track.position = match.centroid
        track.missing = 0
        track.propagated = True
        track.followed_frames += 1
        return True

    def update(
        self, detections: list[Detection], motion: NDArray, frame: Image | None = None
    ) -> list[Track]:
        """Follow established objects and use motion detections to acquire new objects.

        :param detections: Strict moving candidates in current-frame coordinates.
        :param motion: Previous-to-current 2x3 camera transform.
        :param frame: Optional BGR image enabling visual following across detection gaps.
        :return: Active tracks including temporarily missing, unrendered objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        followed: set[int] = set()

        # Predict camera-induced motion first, then ask each established visual model to follow.
        for i, track in enumerate(self.tracks):
            was_missing = track.missing
            position = motion @ np.array([*track.position, 1.0])
            track.position = (float(position[0]), float(position[1]))
            track.missing += 1
            track.propagated = False
            if gray is not None and self._follow(track, frame, gray, was_missing):
                followed.add(i)

        # Candidates within an already-followed object do not create duplicate IDs.
        matched_detections = {
            j
            for j, detection in enumerate(detections)
            if any(self._contains(self.tracks[i].detection, detection.centroid) for i in followed)
        }
        pairs = sorted(
            (float(np.linalg.norm(np.subtract(track.position, detection.centroid))), i, j)
            for i, track in enumerate(self.tracks)
            if i not in followed

            # A confirmed visual track cannot jump to an unrelated motion contour.
            if not (
                gray is not None
                and self.appearance
                and track.confirmed
                and track.follower is not None
            )
            for j, detection in enumerate(detections)
            if j not in matched_detections
        )

        # Greedily associate the nearest eligible detections, using each track and detection once.
        matched_tracks: set[int] = set()
        for distance, i, j in pairs:
            if distance > self.max_distance:
                break
            if i in matched_tracks or j in matched_detections:
                continue
            track = self.tracks[i]
            track.detection = detections[j]
            track.position = detections[j].centroid
            track.consecutive_hits = track.consecutive_hits + 1 if track.missing == 1 else 1
            track.hits += 1
            track.missing = 0
            if track.consecutive_hits >= self.min_hits:
                track.confirmed = True
                if gray is not None and track.follower is None and self.appearance:
                    track.follower = self._new_follower(frame, gray, track.detection)
            matched_tracks.add(i)
            matched_detections.add(j)

        # Break confirmation streaks across gaps and expire tracks beyond the loss tolerance.
        for track in self.tracks:
            if track.missing:
                track.consecutive_hits = 0
        self.tracks = [track for track in self.tracks if track.missing <= self.max_missing]

        # Create tentative identities for remaining detections; confirmation prevents flicker.
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                follower = (
                    self._new_follower(frame, gray, detection)
                    if gray is not None and self.min_hits == 1 and self.appearance
                    else None
                )
                self.tracks.append(
                    Track(
                        self.next_identifier,
                        detection,
                        detection.centroid,
                        confirmed=self.min_hits == 1,
                        follower=follower,
                    )
                )
                self.next_identifier += 1
        self.previous_gray = gray
        return self.tracks

    @staticmethod
    def _contains(detection: Detection, point: tuple[float, float]) -> bool:
        """Return whether a current candidate lies inside an established object box."""
        x, y, width, height = detection.box

        # Use current box membership to suppress detections already covered by a visual track.
        return x <= point[0] < x + width and y <= point[1] < y + height
