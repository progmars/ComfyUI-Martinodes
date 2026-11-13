import unittest

import torch

from martinodes.LatentAVSlicer import LatentAVSlicer
from martinodes.shared import MINIMAX_H3_PARAMS


class LatentAVSlicerTests(unittest.TestCase):
    def setUp(self):
        self.slicer = LatentAVSlicer()
        self.video = torch.arange(12).reshape(1, 1, 12, 1, 1)
        self.audio = torch.arange(80).reshape(1, 1, 80)
        self.latent = {"samples": [self.video, self.audio], "tag": "original"}

    def test_empty_absolute_range_returns_none_before_minimum_duration_alignment(self):
        for offset in (0.0, 17 / 24):
            with self.subTest(offset=offset):
                self.assertIsNone(self.slicer.slice_av_latent(
                    MINIMAX_H3_PARAMS, self.latent, offset, offset, "start"
                ))
        self.assertEqual(self.slicer.run(self.latent, 0, 0, "start"), (None,))

    def test_empty_end_slice_and_audio_slice_return_none(self):
        self.assertIsNone(self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, self.latent, 10, 0, "end"
        ))
        self.assertIsNone(self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, self.latent, 0, 10, "end"
        ))
        self.assertIsNone(self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, self.latent, 2, 0.1, "start"
        ))
        no_audio = {"samples": [self.video, self.audio[..., :0]]}
        self.assertIsNone(self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, no_audio, 0, 0, "end"
        ))

    def test_nonempty_slices_preserve_alignment_and_metadata(self):
        result = self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, self.latent, 17 / 24, 0, "end"
        )
        self.assertEqual(result["tag"], "original")
        self.assertTrue(torch.equal(result["samples"][0], self.video[:, :, 5:]))
        self.assertTrue(torch.equal(result["samples"][1], self.audio[..., 28:]))
        result = self.slicer.slice_av_latent(
            MINIMAX_H3_PARAMS, self.latent, 0, 0.1, "start"
        )
        self.assertEqual(result["samples"][0].shape[2], 2)
        self.assertEqual(result["samples"][1].shape[-1], 8)

    def test_missing_samples_still_raises(self):
        with self.assertRaises(ValueError):
            self.slicer.slice_av_latent(MINIMAX_H3_PARAMS, {}, 0, 0, "start")


if __name__ == "__main__":
    unittest.main()
