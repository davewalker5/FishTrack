"""Optional first-frame rectangle selection with Python's standard Tk interface."""

import base64

import cv2

from fishtrack.detection import Image


def selection_box(
    start: tuple[float, float], end: tuple[float, float], scale: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a drag in preview coordinates to a bounded source-frame rectangle.

    :param start: Drag start in preview pixels.
    :param end: Drag end in preview pixels; either drag direction is supported.
    :param scale: Preview/source dimension ratio, greater than zero.
    :param width: Source width in pixels.
    :param height: Source height in pixels.
    :return: Left, top, width, height in full-resolution source pixels.
    :raises ValueError: If the resulting region is smaller than eight source pixels.
    """
    if scale <= 0:
        raise ValueError("Preview scale must be positive")
    left = max(0, min(width, round(min(start[0], end[0]) / scale)))
    right = max(0, min(width, round(max(start[0], end[0]) / scale)))
    top = max(0, min(height, round(min(start[1], end[1]) / scale)))
    bottom = max(0, min(height, round(max(start[1], end[1]) / scale)))
    if right - left < 8 or bottom - top < 8:
        raise ValueError("Select a region at least eight pixels wide and high")
    return left, top, right - left, bottom - top


def region_at_point(
    regions: list[tuple[int, int, int, int]], point: tuple[float, float]
) -> int | None:
    """Find the smallest selected box containing a source-coordinate click.

    :param regions: Selected rectangles in source coordinates.
    :param point: Click location in source pixels.
    :return: Index to remove, or None when no rectangle contains the point.
    """
    hits = [
        (width * height, index)
        for index, (x, y, width, height) in enumerate(regions)
        if x <= point[0] <= x + width and y <= point[1] <= y + height
    ]
    return min(hits)[1] if hits else None


def select_regions(
    frame: Image, initial_regions: tuple[tuple[int, int, int, int], ...] = ()
) -> tuple[tuple[int, int, int, int], ...]:
    """Let the user drag one or more fish boxes on the first processed frame.

    :param frame: Unannotated BGR frame at the requested start time.
    :param initial_regions: Proposed or previously selected boxes to review and edit.
    :return: Selected boxes in source pixels; Enter accepts and Escape cancels.
    :raises OSError: If Tk is unavailable or cannot open a display.
    :raises ValueError: If the user cancels selection.
    """
    try:
        import tkinter as tk
    except ImportError as error:
        raise OSError("Interactive selection needs Tk; use --roi X Y W H instead") from error
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise OSError("Cannot open selection window; use --roi X Y W H instead") from error
    try:
        root.title("FishTrack — review fish" if initial_regions else "FishTrack — select fish")
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            min(1280, root.winfo_screenwidth() * 0.85) / width,
            min(800, root.winfo_screenheight() * 0.75) / height,
        )
        preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
        success, encoded = cv2.imencode(".png", preview)
        if not success:
            raise OSError("Cannot prepare selection preview")
        photo = tk.PhotoImage(master=root, data=base64.b64encode(encoded.tobytes()))
        status = tk.StringVar(
            master=root,
            value=(
                "Review boxes; right-click inside a box to remove it. Drag to add fish. "
                "Enter: track. Backspace: undo. Escape: cancel."
            ),
        )
        tk.Label(root, textvariable=status).pack()
        canvas = tk.Canvas(
            root, width=preview.shape[1], height=preview.shape[0], highlightthickness=0
        )
        canvas.pack()
        canvas.create_image(0, 0, image=photo, anchor="nw")
        boxes = list(initial_regions)
        rectangles = [
            canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + width) * scale,
                (y + height) * scale,
                outline="yellow",
                width=2,
            )
            for x, y, width, height in boxes
        ]
        start: tuple[float, float] | None = None
        rectangle: int | None = None
        accepted = False

        def begin(event: tk.Event) -> None:
            """Start a selection rectangle at the pointer."""
            nonlocal start, rectangle
            start = (event.x, event.y)
            rectangle = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="yellow", width=2
            )

        def drag(event: tk.Event) -> None:
            """Update the currently dragged rectangle."""
            if start is not None and rectangle is not None:
                canvas.coords(rectangle, *start, event.x, event.y)

        def finish(event: tk.Event) -> None:
            """Accept a valid region or remove a tiny accidental selection."""
            nonlocal start, rectangle
            if start is None or rectangle is None:
                return
            try:
                box = selection_box(start, (event.x, event.y), scale, width, height)
            except ValueError as error:
                canvas.delete(rectangle)
                status.set(str(error))
            else:
                boxes.append(box)
                rectangles.append(rectangle)
                status.set(
                    f"{len(boxes)} fish selected. Enter: track. Backspace: undo. Escape: cancel."
                )
            start, rectangle = None, None

        def undo(event: tk.Event) -> None:
            """Remove the most recently selected fish."""
            if boxes:
                boxes.pop()
                canvas.delete(rectangles.pop())
                status.set(f"{len(boxes)} fish selected. Drag another box or press Enter.")

        def remove(event: tk.Event) -> None:
            """Remove a clicked proposal without disturbing other selected fish."""
            index = region_at_point(boxes, (event.x / scale, event.y / scale))
            if index is not None:
                boxes.pop(index)
                canvas.delete(rectangles.pop(index))
                status.set(
                    f"{len(boxes)} fish selected. "
                    "Drag to add; right-click to remove; Enter to track."
                )

        def accept(event: tk.Event) -> None:
            """Start tracking only after at least one region has been selected."""
            nonlocal accepted
            if boxes:
                accepted = True
                root.quit()
            else:
                status.set("Select at least one fish before pressing Enter.")

        def cancel(event: tk.Event | None = None) -> None:
            """Close the selection workflow without exporting a video."""
            root.quit()

        canvas.bind("<ButtonPress-1>", begin)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", finish)
        canvas.bind("<Button-2>", remove)
        canvas.bind("<Button-3>", remove)
        canvas.bind("<Control-Button-1>", remove)
        root.bind("<BackSpace>", undo)
        root.bind("<Return>", accept)
        root.bind("<Escape>", cancel)
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.lift()
        canvas.focus_set()
        root.mainloop()
        if not accepted:
            raise ValueError("Fish selection cancelled")
        return tuple(boxes)
    except tk.TclError as error:
        raise OSError("Selection window failed; use --roi X Y W H instead") from error
    finally:
        root.destroy()
