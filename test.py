from __future__ import print_function

import time
from collections import deque
from multiprocessing.pool import ThreadPool

import cv2
import cv2 as cv


class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data


if __name__ == '__main__':
    import sys

    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = cv2.VideoCapture(1)


    def process_frame(frame, t0):
        # some intensive computation...
        frame = cv.medianBlur(frame, 19)
        frame = cv.medianBlur(frame, 19)
        return frame, t0


    threadn = cv.getNumberOfCPUs()
    print(threadn)
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = True

    latency = time.time()
    frame_interval = time.time()
    last_frame_time = time.time()
    while True:
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency = time.time() - t
            # draw_str(res, (20, 20), "threaded      :  " +             str(threaded_mode))
            # draw_str(res, (20, 40), "latency        :  %.1f ms" %     (latency.value*1000))
            # draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv.imshow('threaded video', res)
        if len(pending) < threadn:
            ret, frame = cap.read()
            t = time.time()
            frame_interval = t - last_frame_time
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
            pending.append(task)

    cv.destroyAllWindows()
