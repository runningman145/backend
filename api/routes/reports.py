"""
Reports endpoints.
Generates court-ready PDF reports from officer-edited CSV data with optional Mapbox mapping.
"""
import base64
import json
import csv
import hashlib
import hmac
import os
import requests
from datetime import datetime
from io import BytesIO, StringIO
from urllib.parse import quote
from flask import Blueprint, jsonify, request, send_file, current_app
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from ..db import get_db

bp = Blueprint('reports', __name__, url_prefix='/reports')


def _parse_csv_data(csv_text, case_number=None, officer_name=None):
    """
    Parse officer-edited CSV into case metadata and sightings.

    Supports two formats:

    Format A – Two-section (legacy):
        case_number,officer_name,date
        CASE-001,Officer Name,2024-04-28

        sighting_id,timestamp,camera_id,camera_name,include,match_score,officer_note
        1,2024-04-28T09:15:22Z,CAM-001,Main St,true,92.5,Note here

    Format B – Flat (job results CSV exported by the backend):
        time,match_percent,camera_id,camera_name,video_filename,...
        9.0,72.5,...

    When case_number and officer_name are provided as arguments (from FormData)
    they always override anything inside the CSV.

    Returns:
        Tuple of (case_metadata dict, sightings_list)
    """
    lines = csv_text.strip().split('\n')
    if not lines:
        raise ValueError("CSV is empty")

    # Strip carriage returns that Windows line-endings introduce
    lines = [l.rstrip('\r') for l in lines]

    # ------------------------------------------------------------------ #
    # Detect format: if the first header row contains 'case_number' as a  #
    # column name it is Format A; otherwise treat as Format B (flat).     #
    # ------------------------------------------------------------------ #
    first_header = lines[0].lower()
    is_two_section = 'case_number' in first_header and 'officer_name' in first_header

    if is_two_section:
        # ---- Format A ----
        metadata_reader = csv.DictReader(iter(lines[:2]))
        metadata_rows = list(metadata_reader)
        if not metadata_rows:
            raise ValueError("Missing case metadata in CSV")
        case_meta = metadata_rows[0]

        # Find where sightings begin (after empty line)
        sightings_start = None
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 2:
                sightings_start = i + 1
                break
        if sightings_start is None:
            sightings_start = 3

        sightings_text = '\n'.join(lines[sightings_start:])
        sightings_reader = csv.DictReader(StringIO(sightings_text))
        sightings = list(sightings_reader)
    else:
        # ---- Format B (flat job-results CSV) ----
        # Build minimal case metadata from arguments / defaults
        case_meta = {
            'case_number': case_number or 'N/A',
            'officer_name': officer_name or 'N/A',
            'date': datetime.utcnow().strftime('%Y-%m-%d'),
        }
        sightings_reader = csv.DictReader(iter(lines))
        sightings = list(sightings_reader)

        # Normalise column names so the rest of the pipeline works
        # regardless of whether it's Format A or B.
        for s in sightings:
            # match_percent → match_score
            if 'match_percent' in s and 'match_score' not in s:
                s['match_score'] = s.pop('match_percent')
            # time → timestamp (use a synthetic ISO timestamp if needed)
            if 'time' in s and 'timestamp' not in s:
                s['timestamp'] = s.pop('time')
            # Default include to true so all rows appear in the report
            if 'include' not in s:
                s['include'] = 'true'
            # camera_name fallback
            if 'camera_name' not in s:
                s['camera_name'] = s.get('camera_id', 'Unknown')

    # Override case metadata with explicit FormData values when provided
    if case_number:
        case_meta['case_number'] = case_number
    if officer_name:
        case_meta['officer_name'] = officer_name

    # Convert include to boolean and match_score to float
    for sighting in sightings:
        raw_include = sighting.get('include', 'true')
        sighting['include'] = str(raw_include).lower() in ('true', '1', 'yes', '')
        try:
            sighting['match_score'] = float(sighting.get('match_score', 0) or 0)
        except (ValueError, TypeError):
            sighting['match_score'] = 0.0

    return case_meta, sightings


def _fetch_camera_coordinates(camera_ids):
    """
    Fetch camera GPS coordinates from database.

    Returns:
        Dict mapping camera_id to {latitude, longitude}
    """
    db = get_db()
    placeholders = ','.join('?' * len(camera_ids))
    rows = db.execute(
        f'SELECT id, latitude, longitude FROM cameras WHERE id IN ({placeholders})',
        camera_ids
    ).fetchall()

    coords = {}
    for row in rows:
        try:
            lat = float(row['latitude']) if row['latitude'] is not None else None
            lon = float(row['longitude']) if row['longitude'] is not None else None
        except (TypeError, ValueError):
            lat, lon = None, None
        coords[row['id']] = {'latitude': lat, 'longitude': lon}
    return coords


def _fetch_track_sightings(track_id):
    """Fetch ordered sightings for a track; returns list compatible with _generate_mapbox_map."""
    try:
        db = get_db()
        rows = db.execute(
            '''SELECT vd.timestamp, vd.match_score,
                      c.id as camera_id, c.name as camera_name,
                      c.latitude, c.longitude
               FROM track_detections td
               JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               JOIN cameras c ON vd.camera_id = c.id
               WHERE td.track_id = ?
               ORDER BY vd.timestamp ASC''',
            (track_id,)
        ).fetchall()
        sightings = []
        for r in rows:
            try:
                lat = float(r['latitude']) if r['latitude'] is not None else None
                lon = float(r['longitude']) if r['longitude'] is not None else None
            except (TypeError, ValueError):
                lat, lon = None, None
            sightings.append({
                'camera_id': r['camera_id'],
                'camera_name': r['camera_name'],
                'latitude': lat,
                'longitude': lon,
                'timestamp': r['timestamp'],
                'confidence': r['match_score'],
            })
        return sightings
    except Exception as exc:
        current_app.logger.warning(f"Could not fetch sightings for track {track_id}: {exc}")
        return []


def _fetch_frame_images(job_id):
    """
    Return a dict mapping (timestamp_float, camera_id_str) -> base64 data URI
    for every match stored in the job's result_data JSON.
    Returns {} on any failure.
    """
    if not job_id:
        return {}
    try:
        db = get_db()
        row = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        if not row or not row['result_data']:
            return {}
        result_data = json.loads(row['result_data'])
        frame_map = {}
        for match in result_data.get('matches', []):
            ts = match.get('time')
            cam_id = str(match.get('camera_id', ''))
            img = match.get('frame_image')
            if ts is not None and img:
                frame_map[(float(ts), cam_id)] = img
        return frame_map
    except Exception as exc:
        current_app.logger.warning(f"Could not fetch frame images for job {job_id}: {exc}")
        return {}


def _decode_frame_image(data_uri, width_in, height_in):
    """
    Decode a base64 JPEG data URI and return a ReportLab Image flowable,
    or None on any failure.
    """
    if not data_uri:
        return None
    try:
        b64_part = data_uri.split(',', 1)[1] if ',' in data_uri else data_uri
        decoded = base64.b64decode(b64_part)
        return Image(BytesIO(decoded), width=width_in * inch, height=height_in * inch)
    except Exception as exc:
        current_app.logger.warning(f"Could not decode frame image: {exc}")
        return None


def _lookup_frame_image(sighting, frame_map):
    """
    Find the frame image data URI for a sighting.

    Tries (timestamp, camera_id) exact match, then approximate (±0.01 s) for same
    camera, then approximate for any camera (single-camera job fallback).
    """
    if not frame_map:
        return None
    try:
        ts = float(sighting.get('timestamp', ''))
    except (ValueError, TypeError):
        return None
    cam_id = str(sighting.get('camera_id', ''))

    if (ts, cam_id) in frame_map:
        return frame_map[(ts, cam_id)]
    for (fts, fcam), img in frame_map.items():
        if fcam == cam_id and abs(fts - ts) <= 0.01:
            return img
    for (fts, _), img in frame_map.items():
        if abs(fts - ts) <= 0.01:
            return img
    return None


def _ts_sort_key(sighting):
    """
    Sort key for sightings by timestamp.

    Format B CSVs carry float seconds (e.g. "9.0", "12.0") that must be sorted
    numerically — string sort would place "12.0" before "9.0" because '1' < '9'.
    Format A CSVs carry ISO strings ("2024-04-28T09:15:22Z") which sort correctly
    as strings; float() raises ValueError for them so we fall back to string order.
    """
    ts = sighting.get('timestamp', '')
    try:
        return (0, float(ts), '')
    except (ValueError, TypeError):
        return (1, 0.0, str(ts))


def _fetch_detection_timestamps(job_id):
    """
    Return a dict mapping (timestamp_float, camera_id_str) -> wall-clock display string
    by joining vehicle_detections with detections.captured_at.

    Replicates Track Explorer's _fmt_detection_ts logic:
      - vd.timestamp > 1e9  →  absolute UTC epoch  →  "YYYY-MM-DD HH:MM:SS"
      - otherwise           →  detections.captured_at (recording session wall-clock time)

    Returns {} on any failure so callers fall back gracefully.
    """
    if not job_id:
        return {}
    try:
        db = get_db()
        rows = db.execute(
            '''SELECT vd.timestamp, vd.camera_id, d.captured_at
               FROM vehicle_detections vd
               JOIN detections d ON vd.detection_id = d.id
               WHERE vd.job_id = ?''',
            (job_id,)
        ).fetchall()
        ts_map = {}
        for row in rows:
            try:
                t = float(row['timestamp'])
                cam_id = str(row['camera_id'])
                captured_at = row['captured_at']
                if t > 1_000_000_000:
                    from datetime import timezone as _tz
                    display = datetime.fromtimestamp(t, tz=_tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                else:
                    display = captured_at
                if display:
                    ts_map[(t, cam_id)] = str(display)
            except (TypeError, ValueError):
                continue
        return ts_map
    except Exception as exc:
        current_app.logger.warning(f"Could not fetch detection timestamps for job {job_id}: {exc}")
        return {}


def _format_display_ts(sighting, ts_map):
    """
    Return the best available timestamp string for a sighting.

    Tries the wall-clock time from ts_map (same source as Track Explorer) first,
    then falls back to a video-offset string ("1m 30s") derived from the float timestamp.
    """
    if ts_map:
        try:
            ts_float = float(sighting.get('timestamp', ''))
            cam_id = str(sighting.get('camera_id', ''))
            # Exact match
            if (ts_float, cam_id) in ts_map:
                return ts_map[(ts_float, cam_id)]
            # Approximate match ±0.01 s, same camera
            for (fts, fcam), display in ts_map.items():
                if fcam == cam_id and abs(fts - ts_float) <= 0.01:
                    return display
            # Any-camera approximate (single-camera job fallback)
            for (fts, _fcam), display in ts_map.items():
                if abs(fts - ts_float) <= 0.01:
                    return display
        except (ValueError, TypeError):
            pass

    # Fallback: format as video offset
    raw_ts = str(sighting.get('timestamp', ''))
    try:
        sec = float(raw_ts)
        mins = int(sec // 60)
        secs = int(sec % 60)
        return f"{mins}m {secs}s" if mins else f"{secs}s"
    except ValueError:
        return raw_ts[:19]


def _fetch_query_image_data(job_id):
    """
    Return a BytesIO of the job's query image file, or None if unavailable.
    Falls back to the first entry in job_query_images for batch jobs.
    """
    if not job_id:
        return None
    try:
        db = get_db()
        row = db.execute(
            'SELECT query_image_filename FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        filename = row['query_image_filename'] if row else None

        if not filename:
            img_row = db.execute(
                'SELECT query_image_filename FROM job_query_images '
                'WHERE job_id = ? ORDER BY created_at LIMIT 1',
                (job_id,)
            ).fetchone()
            if img_row:
                filename = img_row['query_image_filename']

        if not filename:
            return None

        file_path = os.path.join(current_app.instance_path, 'uploads', filename)
        if not os.path.exists(file_path):
            current_app.logger.warning(f"Query image not found on disk: {file_path}")
            return None

        with open(file_path, 'rb') as fh:
            return BytesIO(fh.read())
    except Exception as exc:
        current_app.logger.warning(f"Could not load query image for job {job_id}: {exc}")
        return None


def _get_road_route(coordinates, mapbox_token):
    """
    Call the Mapbox Directions API to get a road-following route.

    coordinates: list of [lon, lat] pairs (at least 2).
    Returns a list of [lon, lat] pairs following actual roads, or the original
    straight-line coordinates if the API call fails.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coordinates)
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving/{coord_str}"
        f"?geometries=geojson&overview=full&access_token={mapbox_token}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes", [])
        if routes:
            road_coords = routes[0]["geometry"]["coordinates"]
            if len(road_coords) > 1:
                return road_coords
    except Exception as exc:
        current_app.logger.warning(f"Directions API failed, using straight line: {exc}")
    return coordinates


def _generate_mapbox_map(sightings, camera_coords, mapbox_token=None):
    """
    Generate a Mapbox static map image with sighting pins and connecting route.

    Returns BytesIO PNG image data, or None if map cannot be generated.
    """
    if not mapbox_token:
        mapbox_token = current_app.config.get('MAPBOX_ACCESS_TOKEN')

    if not mapbox_token:
        current_app.logger.warning("MAPBOX_ACCESS_TOKEN not configured – skipping map")
        return None

    # Collapse consecutive detections at the same camera into a single stop.
    # The CSV has one row per matched frame, so a vehicle seen 10 times at
    # Camera A then 5 times at Camera B would otherwise produce 15 identical
    # waypoints and make the Directions API loop back on itself.
    unique_stops = []  # list of (camera_id, lon, lat)
    for sighting in sightings:
        camera_id = sighting.get('camera_id')
        if camera_id not in camera_coords:
            continue
        c = camera_coords[camera_id]
        lat, lon = c.get('latitude'), c.get('longitude')
        if lat is None or lon is None:
            continue
        if unique_stops and unique_stops[-1][0] == camera_id:
            continue  # same camera as previous stop — skip duplicate
        unique_stops.append((camera_id, lon, lat))

    if len(unique_stops) < 2:
        current_app.logger.warning("Not enough valid sighting coordinates for map – skipping")
        return None

    n = len(unique_stops)
    coordinates = []
    pins = []
    for i, (_, lon, lat) in enumerate(unique_stops):
        coordinates.append([lon, lat])
        color = 'FF0000' if i == 0 else ('00FF00' if i == n - 1 else '0000FF')
        pins.append(f"pin-s+{color}({lon},{lat})")

    # Get road-following route geometry from Mapbox Directions API.
    # Falls back to straight-line coordinates if the API call fails.
    road_coordinates = _get_road_route(coordinates, mapbox_token)

    # Bounding box derived from the camera pin locations (not the road geometry)
    # so padding is based on the actual camera spread, not route detours.
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Pad by 50% of the span on each side, with a minimum of ~0.008° (~900 m) so
    # tightly-clustered cameras still get a readable street-level view.
    lon_padding = max((max_lon - min_lon) * 0.5, 0.008)
    lat_padding = max((max_lat - min_lat) * 0.5, 0.008)
    bbox = (
        f"[{min_lon - lon_padding:.6f},{min_lat - lat_padding:.6f},"
        f"{max_lon + lon_padding:.6f},{max_lat + lat_padding:.6f}]"
    )

    # GeoJSON LineString overlay — the only path format Mapbox Static Images API
    # reliably supports (raw semicolon-separated coordinates are not valid).
    line_geojson = {
        "type": "Feature",
        "properties": {"stroke": "#0000FF", "stroke-width": 3, "stroke-opacity": 0.8},
        "geometry": {"type": "LineString", "coordinates": road_coordinates},
    }
    path_overlay = f"geojson({quote(json.dumps(line_geojson, separators=(',', ':')), safe='')})"
    overlay_parts = [path_overlay]
    overlay_parts.extend(pins)
    overlay = ",".join(overlay_parts)

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/"
        f"{overlay}/"
        f"{bbox}/800x600@2x"
        f"?access_token={mapbox_token}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        map_image = BytesIO(response.content)
        map_image.seek(0)
        return map_image
    except Exception as exc:
        current_app.logger.warning(f"Mapbox request failed – skipping map: {exc}")
        return None


def _generate_document_signature(data, secret_key=None):
    """Generate HMAC-SHA256 signature for the report data."""
    if isinstance(data, dict):
        data_string = json.dumps(data, sort_keys=True)
    else:
        data_string = str(data)

    data_hash = hashlib.sha256(data_string.encode()).hexdigest()
    key = (secret_key or current_app.config.get('SECRET_KEY', 'dev')).encode()
    signature = hmac.new(key, data_string.encode(), hashlib.sha256).hexdigest()

    return {
        'hash': data_hash,
        'signature': signature,
        'timestamp': datetime.utcnow().isoformat(),
        'algorithm': 'HMAC-SHA256'
    }


def _create_court_ready_pdf(case_metadata, sightings, signature_info=None, camera_coords=None, track_map_image=None, track_id=None, job_id=None):
    """
    Create a court-ready PDF report.

    signature_info can be None – in that case the signature page is omitted (preview mode).
    """
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    elements = []

    frame_map = _fetch_frame_images(job_id)
    ts_map = _fetch_detection_timestamps(job_id)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=20,
        alignment=1
    )

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2d5aa6'),
        spaceAfter=12,
        spaceBefore=12
    )

    # === PAGE 1: COVER PAGE ===
    elements.append(Paragraph('VEHICLE TRACKING REPORT', title_style))
    elements.append(Spacer(1, 0.3 * inch))

    case_data = [
        ['CASE NUMBER:', case_metadata.get('case_number', 'N/A')],
        ['OFFICER:', case_metadata.get('officer_name', 'N/A')],
        ['DATE:', case_metadata.get('date', 'N/A')],
        ['REPORT GENERATED:', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
    ]

    case_table = Table(case_data, colWidths=[1.8 * inch, 3.7 * inch])
    case_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(case_table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph('SUMMARY', heading_style))
    summary_text = (
        f"This report documents {len(sightings)} confirmed vehicle sightings "
        "from the automated Re-ID tracking system."
    )
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.5 * inch))

    # === SIGHTINGS TABLE ===
    elements.append(PageBreak())
    elements.append(Paragraph('VEHICLE SIGHTINGS', heading_style))
    elements.append(Spacer(1, 0.15 * inch))

    FRAME_W = 1.9  # thumbnail display width in inches
    FRAME_H = 1.3  # thumbnail display height in inches

    sightings_data = [['#', 'Detection Time', 'Camera Name', 'Coordinates', 'Confidence', 'Frame']]
    for idx, sighting in enumerate(sightings, 1):
        display_ts = _format_display_ts(sighting, ts_map)

        camera_id = sighting.get('camera_id', '')
        camera_name = sighting.get('camera_name', '')

        coords = "Unknown"
        if camera_coords and camera_id in camera_coords:
            c = camera_coords[camera_id]
            if c.get('latitude') is not None and c.get('longitude') is not None:
                coords = f"{c['latitude']:.5f}, {c['longitude']:.5f}"

        data_uri = _lookup_frame_image(sighting, frame_map)
        frame_cell = _decode_frame_image(data_uri, FRAME_W, FRAME_H)
        if frame_cell is None:
            frame_cell = Paragraph('<i>N/A</i>', styles['Normal'])

        sightings_data.append([
            str(idx),
            display_ts,
            camera_name,
            coords,
            f"{sighting.get('match_score', 0):.1f}%",
            frame_cell,
        ])

    sightings_table = Table(
        sightings_data,
        colWidths=[0.4 * inch, 1.0 * inch, 1.0 * inch, 1.3 * inch, 0.8 * inch, 2.0 * inch]
    )
    sightings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(sightings_table)

    # === HIGHEST CONFIDENCE DETECTIONS BY CAMERA ===
    # Find the single best sighting per camera_id
    cameras_best = {}
    for sighting in sightings:
        cam_id = sighting.get('camera_id', '')
        if cam_id not in cameras_best or sighting.get('match_score', 0.0) > cameras_best[cam_id].get('match_score', 0.0):
            cameras_best[cam_id] = sighting

    if cameras_best:
        elements.append(PageBreak())
        elements.append(Paragraph('HIGHEST CONFIDENCE DETECTIONS BY CAMERA', heading_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Read query image bytes once; create a fresh BytesIO per Image object to avoid seek issues
        query_img_bytes = None
        raw_query = _fetch_query_image_data(job_id)
        if raw_query:
            query_img_bytes = raw_query.read()

        caption_style = ParagraphStyle(
            'Caption', parent=styles['Normal'],
            alignment=1, fontSize=9, textColor=colors.HexColor('#444444'), spaceBefore=4
        )
        cam_heading_style = ParagraphStyle(
            'CamHeading', parent=styles['Heading3'],
            fontSize=11, textColor=colors.HexColor('#1f4788'), spaceBefore=10, spaceAfter=6
        )

        IMG_W = 3.0  # image display width in inches
        IMG_H = 2.1  # image display height in inches

        # Sort cameras by their best score descending
        sorted_bests = sorted(
            cameras_best.values(),
            key=lambda s: s.get('match_score', 0.0),
            reverse=True
        )

        for idx, best in enumerate(sorted_bests):
            cam_id_best = best.get('camera_id', '')
            cam_name_best = best.get('camera_name', cam_id_best or 'Unknown Camera')
            conf_best = best.get('match_score', 0.0)

            display_ts_best = _format_display_ts(best, ts_map)

            coords_best = 'Unknown'
            if camera_coords and cam_id_best in camera_coords:
                c = camera_coords[cam_id_best]
                if c.get('latitude') is not None and c.get('longitude') is not None:
                    coords_best = f"{c['latitude']:.5f}, {c['longitude']:.5f}"

            # Query image — fresh BytesIO each time so ReportLab reads from position 0
            query_cell_img = None
            if query_img_bytes:
                try:
                    query_cell_img = Image(BytesIO(query_img_bytes), width=IMG_W * inch, height=IMG_H * inch)
                except Exception as exc:
                    current_app.logger.warning(f"Could not render query image: {exc}")
            query_cell = query_cell_img or Paragraph('<i>Query image unavailable</i>', styles['Normal'])

            # Best frame for this camera
            best_frame_img = _decode_frame_image(_lookup_frame_image(best, frame_map), IMG_W, IMG_H)
            frame_cell = best_frame_img or Paragraph('<i>Frame unavailable</i>', styles['Normal'])

            comparison_table = Table(
                [
                    [query_cell, frame_cell],
                    [Paragraph('<b>Query Image</b>', caption_style),
                     Paragraph(f'<b>Best Match Frame ({conf_best:.1f}%)</b>', caption_style)],
                ],
                colWidths=[3.25 * inch, 3.25 * inch]
            )
            comparison_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                ('VALIGN', (0, 1), (-1, 1), 'TOP'),
                ('GRID', (0, 0), (-1, 0), 0.5, colors.HexColor('#cccccc')),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
                ('TOPPADDING', (0, 1), (-1, 1), 2),
            ]))

            detail_data = [
                ['Detection Time', 'Camera Name', 'Coordinates', 'Confidence'],
                [display_ts_best, cam_name_best, coords_best, f"{conf_best:.1f}%"],
            ]
            detail_table = Table(
                detail_data,
                colWidths=[1.5 * inch, 1.5 * inch, 2.0 * inch, 1.5 * inch]
            )
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#eaf2ff')),
            ]))

            cam_block = KeepTogether([
                Paragraph(f'Camera: {cam_name_best}', cam_heading_style),
                comparison_table,
                Spacer(1, 0.1 * inch),
                detail_table,
            ])
            elements.append(cam_block)
            if idx < len(sorted_bests) - 1:
                elements.append(Spacer(1, 0.3 * inch))

    # === ALLOCATED TRACK — CAMERA ROUTE (optional) ===
    if track_map_image:
        elements.append(PageBreak())
        elements.append(Paragraph('ALLOCATED TRACK — CAMERA ROUTE', heading_style))
        elements.append(Spacer(1, 0.15 * inch))
        if track_id:
            elements.append(Paragraph(f'<i>Track ID: {track_id}</i>', styles['Italic']))
            elements.append(Spacer(1, 0.1 * inch))
        try:
            elements.append(Image(track_map_image, width=7 * inch, height=5.25 * inch))
        except Exception as e:
            current_app.logger.warning(f"Failed to embed track map image: {e}")
            elements.append(Paragraph('<i>Track map image unavailable</i>', styles['Normal']))
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph(
            '<i>Red pin: First camera sighting | Blue pins: Intermediate cameras | '
            'Green pin: Last camera sighting | Blue line: Vehicle route</i>',
            styles['Italic']
        ))

    # === SIGNATURE PAGE (final report only) ===
    if signature_info:
        elements.append(PageBreak())
        elements.append(Paragraph('DIGITAL SIGNATURE', heading_style))
        elements.append(Spacer(1, 0.2 * inch))

        sig_data = [
            ['Signer:', 'Automated System'],
            ['Timestamp:', signature_info.get('timestamp', 'N/A')],
            ['Algorithm:', signature_info.get('algorithm', 'N/A')],
            ['Document Hash:', signature_info.get('hash', '')[:48] + '...'],
            ['Signature:', signature_info.get('signature', '')[:48] + '...'],
        ]

        sig_table = Table(sig_data, colWidths=[1.5 * inch, 4.5 * inch])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(sig_table)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            '<i>This document has been cryptographically signed. The hash and signature above '
            'can be used to verify document integrity and authenticity.</i>',
            styles['Italic']
        ))

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer


def _extract_csv_and_meta(request_obj):
    """
    Pull the CSV text, case_number, officer_name, and optional track_id out of the request.

    Supports multipart/form-data (file in 'csv' field + extra fields)
    and application/json ({'csv': '...', 'case_number': '...', ...}).

    Returns (csv_text, case_number, officer_name, track_id, job_id) – any may be None.
    """
    csv_text = None
    case_number = None
    officer_name = None
    track_id = None
    job_id = None

    if request_obj.files:
        csv_file = request_obj.files.get('csv')
        if csv_file:
            csv_text = csv_file.read().decode('utf-8')

    # Read extra FormData fields regardless of whether a file was uploaded
    if request_obj.form:
        case_number = request_obj.form.get('case_number') or None
        officer_name = request_obj.form.get('officer_name') or None
        track_id = request_obj.form.get('track_id') or None
        job_id = request_obj.form.get('job_id') or None

    if not csv_text and request_obj.is_json:
        data = request_obj.get_json(force=True, silent=True)
        if data:
            csv_text = data.get('csv')
            case_number = case_number or data.get('case_number')
            officer_name = officer_name or data.get('officer_name')
            track_id = track_id or data.get('track_id')
            job_id = job_id or data.get('job_id')

    return csv_text, case_number, officer_name, track_id, job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/preview', methods=['POST'])
def preview_report():
    """
    Generate a preview PDF from officer-edited CSV.

    Accepts multipart/form-data with:
      - csv:          CSV file (job results or two-section format)
      - case_number:  Case reference number (string)
      - officer_name: Name of the reporting officer (string)

    Returns a PDF binary (no digital signature – preview only).
    """
    try:
        csv_text, case_number, officer_name, track_id, job_id = _extract_csv_and_meta(request)

        if not csv_text:
            return jsonify({'error': 'CSV file or csv field is required'}), 400

        case_metadata, all_sightings = _parse_csv_data(csv_text, case_number, officer_name)

        # For preview include all sightings (include=true *or* not set)
        included = [s for s in all_sightings if s.get('include', True)]
        if not included:
            included = all_sightings  # fall back to all rows

        included.sort(key=_ts_sort_key)

        # Camera GPS coords (used for sightings table and allocated track map)
        camera_ids = list({s.get('camera_id') for s in included if s.get('camera_id')})
        camera_coords = _fetch_camera_coordinates(camera_ids) if camera_ids else {}

        # Track route map (optional — from allocated track)
        track_map_image = None
        if track_id:
            track_sightings = _fetch_track_sightings(track_id)
            if track_sightings:
                track_camera_coords = {
                    s['camera_id']: {'latitude': s['latitude'], 'longitude': s['longitude']}
                    for s in track_sightings
                }
                track_map_image = _generate_mapbox_map(track_sightings, track_camera_coords)

        pdf_buffer = _create_court_ready_pdf(
            case_metadata, included, signature_info=None,
            camera_coords=camera_coords,
            track_map_image=track_map_image,
            track_id=track_id,
            job_id=job_id,
        )

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f'preview_{case_metadata.get("case_number", "report")}.pdf'
        )

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error generating report preview: {e}")
        return jsonify({'error': f'Failed to generate preview: {e}'}), 500


@bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Generate and download a court-ready signed PDF report.

    Accepts multipart/form-data with:
      - csv:          CSV file (job results or two-section format)
      - case_number:  Case reference number (string)
      - officer_name: Name of the reporting officer (string)

    Process:
    1. Parse CSV
    2. Filter sightings where include=true
    3. Fetch GPS coords for each camera (if available)
    4. Generate Mapbox static map with route (if token is configured)
    5. Build multi-page PDF with digital signature
    """
    try:
        csv_text, case_number, officer_name, track_id, job_id = _extract_csv_and_meta(request)

        if not csv_text:
            return jsonify({'error': 'CSV file or csv field is required'}), 400

        case_metadata, all_sightings = _parse_csv_data(csv_text, case_number, officer_name)

        included_sightings = [s for s in all_sightings if s.get('include')]
        if not included_sightings:
            # Fall back to all sightings so the report is never empty
            included_sightings = all_sightings

        if not included_sightings:
            return jsonify({'error': 'CSV contains no sightings data'}), 400

        included_sightings.sort(key=_ts_sort_key)

        # Camera GPS coords (used for sightings table and allocated track map)
        camera_ids = list({s.get('camera_id') for s in included_sightings if s.get('camera_id')})
        camera_coords = _fetch_camera_coordinates(camera_ids) if camera_ids else {}

        # Track route map (optional — from allocated track)
        track_map_image = None
        if track_id:
            track_sightings = _fetch_track_sightings(track_id)
            if track_sightings:
                track_camera_coords = {
                    s['camera_id']: {'latitude': s['latitude'], 'longitude': s['longitude']}
                    for s in track_sightings
                }
                track_map_image = _generate_mapbox_map(track_sightings, track_camera_coords)

        # Digital signature
        signature_data = {
            'case_number': case_metadata.get('case_number'),
            'officer_name': case_metadata.get('officer_name'),
            'sighting_count': len(included_sightings),
        }
        signature_info = _generate_document_signature(signature_data)

        pdf_buffer = _create_court_ready_pdf(
            case_metadata, included_sightings, signature_info=signature_info,
            camera_coords=camera_coords,
            track_map_image=track_map_image,
            track_id=track_id,
            job_id=job_id,
        )

        # Persist the chosen track against the job (best-effort, never blocks PDF download)
        if track_id and job_id:
            try:
                db = get_db()
                db.execute(
                    'UPDATE jobs SET selected_track_id = ? WHERE id = ?',
                    (track_id, job_id)
                )
                db.commit()
            except Exception as _e:
                current_app.logger.warning(f"Could not persist selected_track_id: {_e}")

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'report_{case_metadata.get("case_number", "report")}.pdf'
        )

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error generating report: {e}")
        return jsonify({'error': f'Failed to generate report: {e}'}), 500


@bp.route('/verify-signature', methods=['POST'])
def verify_signature():
    """
    Verify document signature for integrity validation.

    Request body:
    {
        'data': {...},
        'signature': '...',
        'hash': '...'
    }
    """
    try:
        data = request.get_json(force=True, silent=True)

        if not data or 'data' not in data or 'signature' not in data:
            return jsonify({'error': 'data and signature fields required'}), 400

        generated_sig = _generate_document_signature(data['data'])

        is_valid = (
            generated_sig['signature'] == data['signature'] and
            generated_sig['hash'] == data.get('hash', '')
        )

        return jsonify({
            'valid': is_valid,
            'algorithm': 'HMAC-SHA256',
            'message': 'Signature valid' if is_valid else 'Signature invalid or data modified'
        })

    except Exception as e:
        current_app.logger.error(f"Error verifying signature: {e}")
        return jsonify({'error': f'Failed to verify signature: {e}'}), 500
