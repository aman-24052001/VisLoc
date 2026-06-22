"""
VisLoc3D API - exposes the camera projection + ORB localization pipeline
as a small HTTP service.

Why this needs a backend at all, when the rest of the simulator runs
live in the browser: the 6-DOF dynamics, motor mixing, PID controller,
and battery model are pure linear algebra/ODE integration, fully ported
to JS and validated bit-for-bit (see docs/3d/assets/sim3d.js). ORB
feature matching and RANSAC homography fitting are not - they need
OpenCV, which has no practical browser equivalent. This service exists
only to provide that one missing piece: given a real drone pose, render
what its camera would actually see and run the same validated ORB
localizer + nadir-offset correction used throughout this project.

Memory discipline (the explicit ask): the world map and its ~20,000 ORB
keypoints are the only meaningfully-sized state, and they're built once
at process startup, not per-request - exactly how AbsoluteLocalizer was
already designed in Phase 1 of the main project (expensive one-time
keypoint detection, cheap repeated localize() calls). A single global
instance is reused across every request; nothing per-request allocates
more than one camera frame and its descriptors.
"""
from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from visloc.world import generate_world
from visloc.localizer import AbsoluteLocalizer
from visloc3d.camera import CameraIntrinsics, render_view, correct_position_estimate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visloc3d-api")

# Canonical world parameters - identical to every other phase of this
# project, so a fix taken here is directly comparable to everything
# already validated (Phase 1's localizer, the camera integration tests).
WORLD_SEED = 42
WORLD_SIZE = 1800
N_BLOBS = 200
N_ROADS = 14
OUTPUT_SIZE = 220
FOV_REFERENCE_FOOTPRINT = 220
FOV_REFERENCE_Z = 200.0

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Generating world map and building ORB index (one-time startup cost)...")
    world = generate_world(width=WORLD_SIZE, height=WORLD_SIZE, seed=WORLD_SEED,
                            n_blobs=N_BLOBS, n_roads=N_ROADS)
    localizer = AbsoluteLocalizer(world, detector="orb")
    intrinsics = CameraIntrinsics.matching_reference_footprint(
        output_size=OUTPUT_SIZE, footprint_at_z_ref=FOV_REFERENCE_FOOTPRINT, z_ref=FOV_REFERENCE_Z)
    state["world"] = world
    state["localizer"] = localizer
    state["intrinsics"] = intrinsics
    log.info("Startup complete: world=%dx%d, ORB keypoints=%d",
             WORLD_SIZE, WORLD_SIZE, len(localizer.world_kp))
    yield
    state.clear()


app = FastAPI(title="VisLoc3D API", lifespan=lifespan)

allowed_origins = os.environ.get("VISLOC_ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins.split(",") if allowed_origins != "*" else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class LocalizeRequest(BaseModel):
    position: list[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] world position")
    quat: list[float] = Field(..., min_length=4, max_length=4, description="[w, x, y, z] attitude quaternion")
    include_frame: bool = Field(False, description="include the rendered camera frame as base64 PNG")


class LocalizeResponse(BaseModel):
    success: bool
    raw_xy: list[float] | None = None
    corrected_xy: list[float] | None = None
    n_inliers: int = 0
    n_matches: int = 0
    frame_png_base64: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "world_loaded": "world" in state}


@app.get("/api/world_meta")
def world_meta():
    return {
        "world_size": WORLD_SIZE, "seed": WORLD_SEED,
        "n_blobs": N_BLOBS, "n_roads": N_ROADS,
        "n_orb_keypoints": len(state["localizer"].world_kp),
        "output_size": OUTPUT_SIZE,
        "fov_deg": state["intrinsics"].fov_deg,
    }


@app.post("/api/localize", response_model=LocalizeResponse)
def localize(req: LocalizeRequest):
    if "localizer" not in state:
        raise HTTPException(503, "World not yet initialized")

    position = np.array(req.position, dtype=float)
    quat = np.array(req.quat, dtype=float)
    qn = np.linalg.norm(quat)
    if qn < 1e-6:
        raise HTTPException(400, "Quaternion must be non-zero")
    quat = quat / qn

    if not (0 <= position[0] <= WORLD_SIZE and 0 <= position[1] <= WORLD_SIZE):
        raise HTTPException(400, f"Position out of world bounds [0, {WORLD_SIZE}]")

    try:
        frame = render_view(state["world"], position, quat, state["intrinsics"])
    except ValueError as e:
        # camera pointing at/above the horizon - documented, expected case
        return LocalizeResponse(success=False)

    res = state["localizer"].localize(frame)
    if not res.success:
        return LocalizeResponse(success=False, n_matches=res.n_matches, n_inliers=res.n_inliers)

    raw_xy = np.array([res.x, res.y])
    corrected_xy = correct_position_estimate(raw_xy, position[2], quat)

    frame_b64 = None
    if req.include_frame:
        ok, buf = cv2.imencode(".png", frame)
        if ok:
            frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return LocalizeResponse(
        success=True, raw_xy=raw_xy.tolist(), corrected_xy=corrected_xy.tolist(),
        n_inliers=res.n_inliers, n_matches=res.n_matches, frame_png_base64=frame_b64,
    )
