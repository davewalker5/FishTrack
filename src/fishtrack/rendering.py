"""Render annotations on copies of original frames."""

import cv2

from fishtrack.detection import Image
from fishtrack.tracking import Track


def annotate(
    frame: Image,
    tracks: list[Track],
    outline: bool = True,
    track_ids: bool = False,
    bounding_boxes: bool = False,
    min_hits: int = 2,
) -> Image:
    """Draw visible, sufficiently persistent tracks on a frame copy.

    :param frame: Original BGR frame.
    :param tracks: Active temporary identities.
    :param outline: Draw approximate contours.
    :param track_ids: Draw numeric identifiers.
    :param bounding_boxes: Draw diagnostic boxes.
    :param min_hits: Consecutive sightings needed unless already visually confirmed.
    :return: Annotated BGR frame; missing tracks are never rendered.
    """
    result = frame.copy()
    for track in tracks:
        if not track.is_visible(min_hits):
            continue
        color = tuple(64 + (track.identifier * factor) % 192 for factor in (47, 89, 137))
        x, y, width, height = track.detection.box
        if outline:
            cv2.drawContours(result, [track.detection.contour], -1, color, 2)
        if bounding_boxes:
            cv2.rectangle(result, (x, y), (x + width, y + height), color, 1)
        if track_ids:
            cv2.putText(
                result,
                f"{track.identifier}{'*' if track.propagated else ''}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return result
