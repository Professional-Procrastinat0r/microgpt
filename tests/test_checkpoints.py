"""Check exported models can be restored independently of the process cache."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import checkpoints
import inference
import microkernel


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = dict(seed=42, n_layer=1, n_embd=4, n_head=2, block_size=8,
                      num_steps=2, learning_rate=.01, beta1=.85, beta2=.99,
                      eps_adam=1e-8, configuration_id="small")
        cls.checkpoint = microkernel.train(["ana", "ann", "nan"], config, ["a", "n"])

    def test_export_restore_scoring_and_generation(self):
        from expops.storage import VFS
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/"compute.yaml"
            config.write_text("storage:\n  mounts:\n    artefact:\n      type: local\n      source: ./exports\n")
            with VFS.from_config(config) as fs:
                reference = checkpoints.export_model(fs, {
                    "checkpoint": self.checkpoint, "dataset_sha256": "test-data"})
                raw = fs.read_bytes(reference["uri"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), reference["sha256"])
            restored = checkpoints.load_bundle(raw)
            self.assertEqual(microkernel.score(self.checkpoint, ["ana"]), microkernel.score(restored, ["ana"]))
            self.assertEqual(inference.generate(self.checkpoint, 3), inference.generate(restored, 3))

    def test_corrupted_weights_rejected(self):
        checkpoint = json.loads(json.dumps(self.checkpoint))
        checkpoint["weights"]["wte"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "checksum"):
            checkpoints.validate_checkpoint(checkpoint)

    def test_wrong_shape_rejected_even_with_matching_checksum(self):
        checkpoint = json.loads(json.dumps(self.checkpoint))
        checkpoint["weights"]["wte"][0].pop()
        checkpoint["weights_sha256"] = hashlib.sha256(checkpoints.canonical_bytes(checkpoint["weights"])).hexdigest()
        with self.assertRaisesRegex(ValueError, "weight row"):
            checkpoints.validate_checkpoint(checkpoint)

    def test_wrong_kernel_rejected(self):
        with self.assertRaisesRegex(ValueError, "different upstream"):
            checkpoints.load_bundle(json.dumps({"schema": 1, "kind": "microgpt-inference-checkpoint",
                                               "upstream_source_sha256": "wrong", "checkpoint": self.checkpoint}))


if __name__ == "__main__":
    unittest.main()
