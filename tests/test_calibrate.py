"""Tests of the calibration arithmetic with made-up plate boxes. No video, no model."""

import unittest

from spotter.calibrate import Box, dedupe, propose_zone, tiles, width_stats


class TilesTest(unittest.TestCase):
    def test_tiles_cover_the_frame_and_overlap(self):
        boxes = tiles(1920, 1080, size=640, overlap=0.5)
        self.assertIn((0, 0, 1920, 1080), boxes)  # the whole frame is scanned too
        squares = [b for b in boxes if b != (0, 0, 1920, 1080)]
        self.assertTrue(all(b[2] - b[0] == 640 and b[3] - b[1] == 640 for b in squares))
        self.assertTrue(all(0 <= b[0] and b[2] <= 1920 and 0 <= b[1] and b[3] <= 1080 for b in squares))
        self.assertIn((1280, 440, 1920, 1080), squares)  # the far corner is covered
        xs = sorted({b[0] for b in squares})
        self.assertEqual(xs, [0, 320, 640, 960, 1280])  # 50% overlap

    def test_small_frames_get_one_tile(self):
        self.assertEqual(tiles(400, 300, size=640), [(0, 0, 400, 300)])


class BoxesTest(unittest.TestCase):
    def test_overlapping_reads_from_two_tiles_become_one_box(self):
        a = Box(100, 100, 220, 140, 0.8)
        b = Box(104, 102, 224, 142, 0.9)  # the same plate seen from a neighbouring tile
        c = Box(900, 500, 1000, 530, 0.7)
        kept = dedupe([a, b, c])
        self.assertEqual(kept, [b, c])

    def test_width_stats(self):
        boxes = [Box(0, 0, 90, 30, 0.9), Box(0, 0, 120, 40, 0.9), Box(0, 0, 150, 50, 0.9)]
        self.assertEqual(width_stats(boxes), (90, 120, 150))

    def test_proposed_zone_covers_every_box_with_a_margin(self):
        boxes = [Box(800, 500, 920, 540, 0.9), Box(1000, 600, 1120, 640, 0.9)]  # plates 120 x 40
        zone = propose_zone(boxes, 1920, 1080)
        x0, y0, x1, y1 = zone
        self.assertLess(x0 * 1920, 800)
        self.assertGreater(x1 * 1920, 1120)
        self.assertLess(y0 * 1080, 500)
        self.assertGreater(y1 * 1080, 640)
        self.assertAlmostEqual(x0, (800 - 120) / 1920 - 0.005, delta=0.011)  # one plate width sideways
        self.assertAlmostEqual(y0, (500 - 80) / 1080 - 0.005, delta=0.011)  # two plate heights up
        self.assertTrue(all(0 <= v <= 1 for v in zone))
        self.assertTrue(all(round(v, 2) == v for v in zone))

    def test_proposed_zone_is_clamped_to_the_frame(self):
        boxes = [Box(0, 0, 200, 60, 0.9), Box(1800, 1030, 1920, 1080, 0.9)]
        self.assertEqual(propose_zone(boxes, 1920, 1080), (0.0, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
