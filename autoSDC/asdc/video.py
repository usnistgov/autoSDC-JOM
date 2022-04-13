import cv2
import typing
import skvideo.io
import numpy as np
import multiprocessing as mp
from contextlib import contextmanager

FFMPEG_OUTPUT_FLAGS = {
    "-vcodec": "libx264",
    "-b": "300000000",
    "-crf": "18",
    "-vf": "format=yuv420p",
}


def videocap(e, filename="testproc.mp4", camera_idx=0):

    cap = cv2.VideoCapture(camera_idx)

    videostream = skvideo.io.FFmpegWriter(filename, outputdict=FFMPEG_OUTPUT_FLAGS)

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        videostream.writeFrame(gray)

        if e.is_set():
            break

    cap.release()
    videostream.close()


@contextmanager
def ffmpeg_capture(filename="testproc.mp4", camera_idx=0):

    e = mp.Event()
    p = mp.Process(target=videocap, args=(e, filename, camera_idx))

    try:
        p.start()
        yield

    finally:
        e.set()
        p.join()


if __name__ == "__main__":

    with ffmpeg_capture(filename="testproc.mp4"):
        input("press enter to wrap.")
