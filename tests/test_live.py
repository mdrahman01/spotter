"""Tests of the live-source pieces: watched time across gaps, camera outages, the masked address,
and the newest-frame reader on a tiny synthetic video."""

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from spotter.events import ArriveLeaveRule
from spotter.live import CameraMonitor, FrameSource, WatchedClock, mask_source

PLATE = "TEST123"


class WatchedClockTest(unittest.TestCase):
    def test_short_gaps_count_and_long_gaps_do_not(self):
        clock = WatchedClock(gap_limit_s=2)
        self.assertEqual(clock.advance(100.0), 0.0)
        self.assertAlmostEqual(clock.advance(100.2), 0.2)
        self.assertAlmostEqual(clock.advance(102.2), 2.2)  # exactly 2 s still counts
        self.assertAlmostEqual(clock.advance(117.2), 2.2)  # a 15 s outage adds nothing
        self.assertAlmostEqual(clock.advance(117.4), 2.4)

    def test_an_outage_is_not_a_departure(self):
        rule = ArriveLeaveRule(arrive_reads=5, arrive_window_s=3, leave_after_s=4)
        clock = WatchedClock()
        real = 1000.0
        for _ in range(10):  # 2 s of reads: the car arrives
            rule.update(clock.advance(real), [PLATE])
            real += 0.2
        self.assertIn(PLATE, rule.present)
        real += 15  # the camera is away for 15 s; the first frame back still shows the car
        self.assertEqual(rule.update(clock.advance(real), [PLATE]), [])
        for _ in range(25):  # then 5 s of frames without the plate: 4 s of watched time pass
            real += 0.2
            events = rule.update(clock.advance(real), [])
            if events:
                break
        self.assertEqual([e.type for e in events], ["LEFT"])
        self.assertAlmostEqual(real - 1017.0, 4.0, delta=0.21)


class CameraMonitorTest(unittest.TestCase):
    def test_offline_once_then_back_once_with_the_downtime(self):
        monitor = CameraMonitor(offline_after_s=10)
        monitor.start(0.0)
        self.assertIsNone(monitor.frame(1.0))
        self.assertFalse(monitor.tick(5.0))
        self.assertFalse(monitor.tick(10.9))
        self.assertTrue(monitor.tick(11.0))  # 10 s after the last frame
        self.assertFalse(monitor.tick(12.0))  # reported once
        self.assertFalse(monitor.tick(30.0))
        self.assertAlmostEqual(monitor.frame(16.0), 15.0)  # down since the last frame at 1.0
        self.assertIsNone(monitor.frame(16.2))
        self.assertEqual(monitor.outages, 1)

    def test_a_source_that_never_opens_counts_as_offline_from_the_start(self):
        monitor = CameraMonitor(offline_after_s=10)
        monitor.start(100.0)
        self.assertFalse(monitor.tick(109.0))
        self.assertTrue(monitor.tick(110.0))
        self.assertAlmostEqual(monitor.frame(130.0), 30.0)


class MaskSourceTest(unittest.TestCase):
    def test_credentials_are_hidden(self):
        self.assertEqual(mask_source("rtsp://admin:s3cret@192.168.1.50:554/stream1"), "rtsp://***@192.168.1.50:554/stream1")
        self.assertEqual(mask_source("rtsp://192.168.1.50/stream1?token=abc"), "rtsp://192.168.1.50/stream1")
        self.assertEqual(mask_source("samples/clip.mov"), "samples/clip.mov")


def write_clip(path: Path, frames: int, fps: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class FrameSourceTest(unittest.TestCase):
    def test_a_file_plays_in_real_time_and_keeps_only_the_newest_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            write_clip(clip, frames=10, fps=10)  # one second of video
            source = FrameSource(str(clip))
            source.start()
            started = time.time()
            deadline = started + 5
            while source.newest() is None and time.time() < deadline:
                time.sleep(0.02)
            first = source.newest()
            self.assertIsNotNone(first)
            time.sleep(0.45)
            later = source.newest()
            self.assertGreater(later[2], first[2])  # newer frames replaced the old one
            self.assertLess(later[2] - first[2], 10)  # paced, not dumped at once
            while not source.ended and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(source.ended)
            self.assertGreaterEqual(time.time() - started, 0.8)  # a one-second file took about a second
            self.assertEqual((source.width, source.height), (64, 48))
            source.stop()

    def test_hold_withholds_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            write_clip(clip, frames=20, fps=10)
            source = FrameSource(str(clip), hold=lambda now: True)
            source.start()
            time.sleep(0.5)
            self.assertIsNone(source.newest())
            self.assertFalse(source.ended)
            source.stop()

    def test_a_missing_file_keeps_retrying_without_crashing(self):
        source = FrameSource("/nonexistent/clip.mp4")
        source.start()
        time.sleep(1.3)
        self.assertGreaterEqual(source.failed_opens, 1)
        self.assertFalse(source.connected)
        source.stop()


if __name__ == "__main__":
    unittest.main()
