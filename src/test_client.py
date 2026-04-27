"""Bare-bones MAVC test client.

Connects to a MAVC-Receiver TCP server (e.g. the one started by
``src/run.py``) and either sends one Command and exits, or streams a small
cycle of camera-frame poses at a fixed rate.

This script intentionally has zero IsaacLab / IsaacSim dependencies -- it only
needs the ``mavc_receiver`` package (for ``Command`` + ``CommandParser``).

Usage::

    python test_client.py                        # cycle 3 poses at 20 Hz, ~4 s each
    python test_client.py --once                 # send DEFAULT_POSES[0] once and exit
    python test_client.py --host 127.0.0.1 --port 9000

Pair with ``run.py``'s defaults: it listens on ``0.0.0.0:9000`` and rescales
``palm_position`` by ``--mavc_reach`` (default ``0.6 m``).
"""

import argparse
import socket
import time
from typing import Tuple

from mavc_receiver import Command, CommandParser

# Wire constants -- must match the receiver's expectations.
MAGIC = 0x073CD
VERSION = 1

# MediaPipe-style camera-frame poses: each row is (px, py, pz, roll, pitch, yaw, grip).
# ``palm_position`` uses raw MediaPipe pose-landmark units -- ``px``/``py`` are
# normalized image coords with the image's top-left as origin (so both are in
# ``[0, 1]`` and the optical axis is at ``(0.5, 0.5)``), and ``pz`` is signed
# depth. The receiver re-centers ``(px, py)`` by ``(-0.5, -0.5)`` and then
# multiplies all three components by ``--mavc_reach`` (default 0.6 m). With
# reach=0.6, the poses below map to camera-frame meters roughly
# ``(+0.24, -0.24, +/-0.24)`` and ``(+0.18, -0.30, 0)``; after the camera->root
# axis swap in utils.transforms (cam X->root X, cam Y->root -Z, cam Z->root -Y)
# they land near root-frame ``(0.24, +/-0.24, +0.24)`` and ``(0.18, 0, +0.30)``
# -- inside the Panda's typical workspace.
DEFAULT_POSES: list[Tuple[float, float, float, float, float, float, float]] = [
    (0.90, 0.10,  0.40, 0.0, 0.0, 0.0, 0.0),
    (0.90, 0.10, -0.40, 0.0, 0.0, 0.0, 0.0),
    (0.80, 0.00,  0.00, 0.0, 0.0, 0.0, 0.5),
]


def make_command(
    seq: int,
    palm_pos: Tuple[float, float, float],
    palm_orient: Tuple[float, float, float],
    grip: float,
) -> Command:
    return Command(
        magic=MAGIC,
        version=VERSION,
        sequence_id=seq,
        timestamp=time.time(),
        palm_position=palm_pos,
        palm_orientation=palm_orient,
        grip_amount=grip,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Bare-bones MAVC test client.")
    ap.add_argument("--host", default="127.0.0.1", help="Receiver host (default: 127.0.0.1).")
    ap.add_argument("--port", type=int, default=9000, help="Receiver port (default: 9000).")
    ap.add_argument("--rate_hz", type=float, default=20.0, help="Send rate when cycling (default: 20).")
    ap.add_argument(
        "--hold_s",
        type=float,
        default=4.0,
        help="Seconds to hold each pose before advancing (cycle mode, default: 4).",
    )
    ap.add_argument("--once", action="store_true", help="Send DEFAULT_POSES[0] once and exit.")
    args = ap.parse_args()

    encoder = CommandParser()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[client] connected to {args.host}:{args.port}")

    try:
        if args.once:
            px, py, pz, r, p, y, g = DEFAULT_POSES[0]
            cmd = make_command(0, (px, py, pz), (r, p, y), g)
            sock.sendall(encoder.encode(cmd))
            print(
                f"[client] sent one frame: pos=({px:+.2f},{py:+.2f},{pz:+.2f}) "
                f"rpy=({r:+.2f},{p:+.2f},{y:+.2f}) grip={g:.2f}"
            )
            return

        period = 1.0 / max(args.rate_hz, 1e-3)
        log_every = max(1, int(args.rate_hz))
        seq = 0
        start = time.monotonic()
        print(f"[client] cycling {len(DEFAULT_POSES)} poses at {args.rate_hz:.1f} Hz, {args.hold_s:.1f} s each")
        while True:
            elapsed = time.monotonic() - start
            idx = int(elapsed // args.hold_s) % len(DEFAULT_POSES)
            px, py, pz, r, p, y, g = DEFAULT_POSES[idx]
            cmd = make_command(seq, (px, py, pz), (r, p, y), g)
            sock.sendall(encoder.encode(cmd))
            if seq % log_every == 0:
                print(
                    f"[client] seq={seq} pose#{idx} pos=({px:+.2f},{py:+.2f},{pz:+.2f}) "
                    f"rpy=({r:+.2f},{p:+.2f},{y:+.2f}) grip={g:.2f}"
                )
            seq += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("[client] interrupted")
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"[client] connection lost: {type(e).__name__}: {e}")
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


if __name__ == "__main__":
    main()
