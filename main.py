"""
main.py
────────
Primary entry point for the Store Intelligence System.

Modes
─────
  python main.py server           → FastAPI only (no vision pipeline)
  python main.py pipeline         → Vision pipeline only (no API server)
  python main.py all              → API + vision pipeline concurrently (default)
  python main.py demo             → Process a single frame from webcam and exit
  python main.py demo --source path/to/video.mp4

Usage
─────
  # Development (webcam):
  python main.py all

  # Production (RTSP):
  SIS_RTSP_URL=rtsp://192.168.1.100:554/stream python main.py all

  # API only (pipeline running separately):
  python main.py server
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sis.main")


def parse_args():
    p = argparse.ArgumentParser(description="Store Intelligence System")
    p.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "server", "pipeline", "demo"],
        help="Run mode (default: all)",
    )
    p.add_argument("--source", default=None, help="Video source override (path or RTSP URL)")
    p.add_argument("--mock-alerts", action="store_true", default=True,
                   help="Use mock Twilio alerts (no real calls)")
    p.add_argument("--no-mock-alerts", dest="mock_alerts", action="store_false")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    return p.parse_args()


async def run_api(host: str, port: int) -> None:
    """Start the FastAPI/uvicorn server."""
    config = uvicorn.Config(
        "src.api.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_pipeline(source=None, mock_alerts=True) -> None:
    """Start the video ingestion + inference pipeline."""
    from pipeline.orchestrator import StoreIntelligenceOrchestrator
    from src.api.connection_manager import ws_manager

    orchestrator = StoreIntelligenceOrchestrator(
        mock_alerts=mock_alerts,
        broadcast_fn=ws_manager.broadcast_to_store,
    )
    await orchestrator.run(source=source)


async def run_all(host, port, source, mock_alerts) -> None:
    """Run API server and pipeline concurrently."""
    api_task      = asyncio.create_task(run_api(host, port))
    pipeline_task = asyncio.create_task(run_pipeline(source, mock_alerts))
    logger.info("Both API (%s:%d) and pipeline started", host, port)
    try:
        await asyncio.gather(api_task, pipeline_task)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        api_task.cancel()
        pipeline_task.cancel()


def run_demo(source=None) -> None:
    """Quick sanity check: process 30 frames and print stats."""
    import cv2
    from pipeline.orchestrator import StoreIntelligenceOrchestrator

    orch = StoreIntelligenceOrchestrator(mock_alerts=True)

    src = source or "0"
    try:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open source: %s", src)
        return

    print(f"\n{'─'*60}")
    print("  Store Intelligence System — Demo Mode")
    print(f"  Source : {src}")
    print(f"{'─'*60}\n")

    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        result = orch.process_single_frame(frame)
        print(
            f"  Frame {i+1:3d} | people={result['person_count']:2d} | "
            f"motion={result['has_motion']} | "
            f"inference={result['inference_ms']:.1f}ms"
        )

    cap.release()
    print(f"\n  Final stats: {orch.stats}")
    print(f"\n{'─'*60}\n")


def main() -> None:
    from config.settings import settings

    args = parse_args()
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    if args.mode == "demo":
        run_demo(source=args.source)

    elif args.mode == "server":
        asyncio.run(run_api(host, port))

    elif args.mode == "pipeline":
        asyncio.run(run_pipeline(source=args.source, mock_alerts=args.mock_alerts))

    else:  # "all"
        try:
            asyncio.run(run_all(host, port, args.source, args.mock_alerts))
        except KeyboardInterrupt:
            logger.info("Interrupted — goodbye")


if __name__ == "__main__":
    main()
