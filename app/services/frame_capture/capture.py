import os
from collections import deque
from typing import Optional

import cv2


class FrameCapture:
    def __init__(
        self,
        source: str = "0",
        width: Optional[int] = None,
        height: Optional[int] = None,
        buffer_size: int = 30,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self._camera = None
        self._frame_buffer = deque(maxlen=buffer_size)

    def _parse_source(self):
        if isinstance(self.source, int):
            return self.source
        if isinstance(self.source, str) and self.source.isdigit():
            return int(self.source)
        return self.source

    def open(self) -> None:
        if self._camera is not None and self._camera.isOpened():
            return

        self._camera = cv2.VideoCapture(self._parse_source())
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if not self._camera.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

    def _preprocess(self, frame):
        if self.width and self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def read(self):
        self.open()
        success, frame = self._camera.read()

        if not success:
            parsed_source = self._parse_source()
            if isinstance(parsed_source, str) and os.path.isfile(parsed_source):
                self._camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self._camera.read()

        if not success:
            return None

        processed = self._preprocess(frame)
        self._frame_buffer.append(processed)
        return processed

    def release(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
