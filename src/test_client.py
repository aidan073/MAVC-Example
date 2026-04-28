"""Bare-bones MAVC test client.

Connects to a MAVC-Receiver TCP server (e.g. the one started by
``src/run.py``) and either sends one Command and exits, or streams a small
cycle of shoulder-frame poses at a fixed rate.

This script intentionally has zero IsaacLab / IsaacSim dependencies -- it only
needs the ``mavc_receiver`` package (for ``Command`` + ``CommandParser``).

Usage::

    python test_client.py                        # cycle 3 poses at 20 Hz, ~4 s each
    python test_client.py --once                 # send DEFAULT_POSES[0] once and exit
    python test_client.py --host 127.0.0.1 --port 9000

Pair with ``run.py``'s defaults: it listens on ``0.0.0.0:9000`` and rescales
``palm_position`` by ``--mavc_reach`` (default ``0.6 m``). For the demo poses
below to land inside the Panda's workspace, also pass a non-zero
``--shoulder_xyz`` to ``run.py``, e.g.::

    ./isaaclab.sh -p src/run.py --shoulder_xyz 0.5 0.3 0.4
"""

import argparse
import socket
import time
from typing import Tuple

from mavc_receiver import Command, CommandParser

# Wire constants -- must match the receiver's expectations.
MAGIC = 0x073CD
VERSION = 1

# Shoulder-frame normalized poses: each row is (px, py, pz, roll, pitch, yaw, grip).
# ``palm_position`` is the wrist's position relative to the operator's shoulder,
# expressed in the shoulder frame (which shares orientation with the camera
# frame). Components are normalized; the receiver multiplies them by
# ``--mavc_reach`` (default 0.6 m). With reach=0.6, the poses below map to
# shoulder-frame meters roughly ``(+/-0.24, 0, +0.24)``; after the
# shoulder->root axis swap in utils.transforms (X->X, Y->-Z, Z->-Y) they land
# at root-frame offsets ``(+/-0.24, -0.24, 0)`` from whatever ``--shoulder_xyz``
# is passed to ``run.py``. With ``--shoulder_xyz 0.5 0.3 0.4`` they land near
# ``(0.26..0.74, 0.06, 0.4)`` -- inside the Panda's reach envelope.
DEFAULT_POSES: list[Tuple[float, float, float, float, float, float, float]] = [
    (1.00, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0),
    (-0.40, 0.00, +0.40, 0.0, 0.0, 0.0, 0.0),
    ( 0.00, 0.00, +0.40, 0.0, 0.0, 0.0, 0.5),
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
