from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
import unittest

from preflight_solar_direct_arm_transport_v104 import merged, native_reference_png


class DirectTransportGuards(unittest.TestCase):
    def test_conflicting_source_input_identity_is_never_overwritten(self):
        self.assertEqual(merged({"a": "first"}, {"a": "first", "b": "second"}),
                         {"a": "first", "b": "second"})
        with self.assertRaisesRegex(ValueError, "Conflicting immutable transport pin"):
            merged({"a": "first"}, {"a": "different"})

    def test_native_reference_header_rejects_substitute_content(self):
        prefix = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            for width, depth, mode in ((512, 8, 0), (512, 8, 2)):
                path.write_bytes(prefix + pack(">IIBBBBB", width, 512, depth, mode, 0, 0, 0))
                native_reference_png(path)
            for width, depth, mode in ((504, 8, 0), (512, 16, 0), (512, 8, 3)):
                path.write_bytes(prefix + pack(">IIBBBBB", width, 512, depth, mode, 0, 0, 0))
                with self.assertRaisesRegex(ValueError, "native512 uint8"):
                    native_reference_png(path)
            path.write_bytes(prefix)
            with self.assertRaisesRegex(ValueError, "not a PNG"):
                native_reference_png(path)


if __name__ == "__main__":
    unittest.main()
