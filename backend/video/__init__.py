"""Video capture, pipeline orchestration and overlay rendering.

Intentionally free of re-exports. `pipeline` depends on the event engine,
which depends on `evidence`, which depends on `ring_buffer` — re-exporting
those names here would make importing any one of them execute the whole graph
and deadlock on a partially-initialised module. Import the specific module you
need instead:

    from backend.video.manager import get_manager
    from backend.video.source import VideoSource
    from backend.video.ring_buffer import FrameRingBuffer
"""
