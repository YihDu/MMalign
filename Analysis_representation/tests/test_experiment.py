import numpy as np
from data.schemas import CaptionSample, ImageSample, MultilingualExample, SampleBatch
from experiment import build_conditions, run_experiment
from models.embedding import (
    EmbeddingBatch,
    LanguageFusionEmbedding,
    MultilayerEmbedding,
    SequenceSpanInfo,
)


class DummyModel:
    def get_image_features(self, *, images):
        class Output:
            def __init__(self, count: int) -> None:
                self._values = np.zeros((count, 2))

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        return Output(len(images))

    def get_text_features(self, *, text):
        class Output:
            def __init__(self, count: int) -> None:
                self._values = np.zeros((count, 2))

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        return Output(len(text))


class DummyProcessor:
    def __call__(self, *, images=None, text=None, **_kwargs):
        if images is not None:
            return {"images": images}
        if text is not None:
            return {"text": text}
        return {}


def make_sample_batch() -> SampleBatch:
    image = ImageSample(image_id=1, image_data="image-bytes")
    captions = {
        "en": CaptionSample(language="en", text="A cat on a mat"),
        "zh": CaptionSample(language="zh", text="一只猫在垫子上"),
    }
    example = MultilingualExample(image=image, captions=captions)
    return SampleBatch([example])


def test_runner_produces_results(monkeypatch):
    batch = make_sample_batch()
    conditions = build_conditions()

    def fake_encode_examples(examples, *_args, **_kwargs):
        example_list = list(examples)
        count = len(example_list)
        per_layer = [np.zeros((count, 2))]
        embedding = MultilayerEmbedding(per_layer=per_layer, pooled=per_layer[-1])
        span_info = SequenceSpanInfo(
            fused_lengths=[2] * count,
            image_spans=[(0, 1)] * count,
            text_last_indices=[1] * count,
            num_image_tokens=1,
        )
        language_embedding = LanguageFusionEmbedding(
            text=embedding,
            image=embedding,
            text_only=embedding,
            delta_text=embedding,
            spans=span_info,
        )
        captions = {"en": language_embedding, "zh": language_embedding}
        return EmbeddingBatch(captions=captions)

    monkeypatch.setattr("experiment.runner.encode_examples", fake_encode_examples)

    distance_map = run_experiment(
        model=DummyModel(),
        processor=DummyProcessor(),
        batch=batch,
        conditions=conditions,
        languages=["en", "zh"],
    )
    assert "correct" in distance_map
    assert "mismatched" in distance_map
