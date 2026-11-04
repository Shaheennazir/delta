"""Offline tests for the real SAM path: wire grammar -> parser -> geometry.

Uses verbatim payload shapes from the meta-sam-parser README (valid
arithmetic-coded masks) fed as mapping events, plus an imageio-encoded MP4
for the media module. No API key needed.
"""
import io

import imageio.v3 as iio
import numpy as np
import pytest

# Canonical one_bit payload from the meta-sam conformance suite
# (synthetic-video-basic): 3x2 raster [1,0,1,0,1,0]. Boxes match the raster.
LINE0 = (
    "<0f>0<|box;x1=5;y1=8;x2=7;y2=9;w=20;h=10|>"
    "<|mask;x=0;y=0;data=2,3,!!!!!&[{Vkw|>"
    ",1<|box;x1=12;y1=4;x2=14;y2=5;w=20;h=10|>"
    "<|mask;x=0;y=0;data=2,3,!!!!!&[{Vkw|>\n"
)
LINE1 = (
    "<1f>0<|box;x1=6;y1=8;x2=8;y2=9;w=20;h=10|>"
    "<|mask;x=0;y=0;data=2,3,!!!!!&[{Vkw|>"
    ",1<|box;x1=12;y1=4;x2=14;y2=5;w=20;h=10|>"
    "<|mask;x=0;y=0;data=2,3,!!!!!&[{Vkw|>\n"
)


async def _feed(lines: str):
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    yield {"type": "response.output_text.delta", "delta": lines, **lane}
    yield {"type": "response.output_text.done", "text": lines, **lane}


def _parse(lines: str):
    import asyncio

    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    async def go():
        return await parse_responses_stream(_feed(lines), video_segmentation_format()).final_result()

    return asyncio.run(go())


def test_wire_lane_parses_to_mask_records():
    result = _parse(LINE0 + LINE1)
    masks = [r for r in result.records if r.kind == "mask"]
    assert len(masks) == 4  # 2 objects x 2 frames
    assert {m.object_id for m in masks} == {"0", "1"}
    assert masks[0].mask.width == 3 and masks[0].mask.height == 2
    assert not result.diagnostics


def test_tracks_from_result_offsets_to_media_pixels():
    from sam_client import tracks_from_result

    result = _parse(LINE0 + LINE1)
    parsed = tracks_from_result(result, 2, media=(20, 10))
    assert parsed["primary_id"] in ("0", "1")
    foot = parsed["objects"]["0"]["foot_px"]
    assert foot.shape == (2, 2) and np.isfinite(foot).all()
    # Foot in media coords: raster-local (1,1) + box origin; obj 0 moves +1px.
    assert tuple(foot[0]) == (6.0, 9.0)
    assert tuple(foot[1]) == (7.0, 9.0)
    assert parsed["media"] == (20, 10)
    assert parsed["objects"]["0"]["area_px"][0] == 3.0


def test_media_module_roundtrip():
    from media import frame_to_jpeg, probe, read_frame, validate_upload

    frames = [np.full((128, 160, 3), i * 25, dtype=np.uint8) for i in range(10)]
    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(frames), extension=".mp4", fps=10)
    data = buf.getvalue()
    meta = validate_upload(data)
    assert meta["width"] == 160 and meta["height"] == 128
    assert abs(meta["fps"] - 10) < 1.0
    f0 = read_frame(data, 0)
    assert f0.shape == (128, 160, 3)
    assert frame_to_jpeg(f0)[:2] == b"\xff\xd8"
    with pytest.raises(ValueError):
        validate_upload(b"not a video")
