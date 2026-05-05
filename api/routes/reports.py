"""
Reports endpoints.
Generates court-ready PDF reports from officer-edited CSV data with optional Mapbox mapping.
"""
import json
import csv
import hashlib
import hmac
import requests
from datetime import datetime
from io import BytesIO, StringIO
from flask import Blueprint, jsonify, request, send_file, current_app
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image
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
        coords[row['id']] = {
            'latitude': row['latitude'],
            'longitude': row['longitude']
        }
    return coords


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

    coordinates = []
    pins = []

    for idx, sighting in enumerate(sightings):
        camera_id = sighting.get('camera_id')
        if camera_id not in camera_coords:
            continue
        coords = camera_coords[camera_id]
        lat = coords['latitude']
        lon = coords['longitude']
        coordinates.append([lon, lat])
        color = 'FF0000' if idx == 0 else ('00FF00' if idx == len(sightings) - 1 else '0000FF')
        pins.append(f"pin-s+{color}({lon},{lat})")

    if len(coordinates) < 2:
        current_app.logger.warning("Not enough valid sighting coordinates for map – skipping")
        return None

    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_padding = (max_lon - min_lon) * 0.1 or 0.01
    lat_padding = (max_lat - min_lat) * 0.1 or 0.01

    polyline = ";".join(f"{lon},{lat}" for lon, lat in coordinates)
    overlay_parts = [f"path-3+0000FF-0.5({polyline})"]
    overlay_parts.extend(pins)
    overlay = "/".join(overlay_parts)

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/"
        f"{overlay}/"
        f"auto/800x600@2x"
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


def _create_court_ready_pdf(case_metadata, sightings, map_image, signature_info=None):
    """
    Create a court-ready PDF report.

    map_image can be None – in that case the map page is omitted.
    signature_info can be None – in that case the signature page is omitted (preview mode).
    """
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    elements = []

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

    # === PAGE 2: MAP (optional) ===
    if map_image:
        elements.append(PageBreak())
        elements.append(Paragraph('VEHICLE ROUTE MAP', heading_style))
        elements.append(Spacer(1, 0.15 * inch))
        try:
            img = Image(map_image, width=7 * inch, height=5.25 * inch)
            elements.append(img)
        except Exception as e:
            current_app.logger.warning(f"Failed to embed map image: {e}")
            elements.append(Paragraph('<i>Map image unavailable</i>', styles['Normal']))
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph(
            '<i>Red pin: First sighting | Blue pins: Intermediate sightings | '
            'Green pin: Last sighting | Blue line: Route in chronological order</i>',
            styles['Italic']
        ))

    # === SIGHTINGS TABLE ===
    elements.append(PageBreak())
    elements.append(Paragraph('VEHICLE SIGHTINGS', heading_style))
    elements.append(Spacer(1, 0.15 * inch))

    sightings_data = [['#', 'Time / Offset', 'Camera ID', 'Location', 'Confidence', 'Officer Notes']]
    for idx, sighting in enumerate(sightings, 1):
        # timestamp may be a raw seconds float in the flat format
        raw_ts = str(sighting.get('timestamp', ''))
        try:
            sec = float(raw_ts)
            mins = int(sec // 60)
            secs = int(sec % 60)
            display_ts = f"{mins}m {secs}s" if mins else f"{secs}s"
        except ValueError:
            display_ts = raw_ts[:19]  # ISO timestamp, strip microseconds/Z

        sightings_data.append([
            str(idx),
            display_ts,
            sighting.get('camera_id', ''),
            sighting.get('camera_name', ''),
            f"{sighting.get('match_score', 0):.1f}%",
            sighting.get('officer_note', '') or sighting.get('officer_notes', ''),
        ])

    sightings_table = Table(
        sightings_data,
        colWidths=[0.5 * inch, 1.2 * inch, 0.9 * inch, 1.5 * inch, 0.9 * inch, 1.5 * inch]
    )
    sightings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(sightings_table)

    # === OFFICER NOTES PAGE ===
    elements.append(PageBreak())
    elements.append(Paragraph('OFFICER NOTES', heading_style))
    elements.append(Spacer(1, 0.2 * inch))

    notes_by_sighting = []
    for sighting in sightings:
        note = sighting.get('officer_note', '') or sighting.get('officer_notes', '')
        if note:
            raw_ts = str(sighting.get('timestamp', ''))
            try:
                sec = float(raw_ts)
                mins = int(sec // 60)
                secs = int(sec % 60)
                display_ts = f"{mins}m {secs}s" if mins else f"{secs}s"
            except ValueError:
                display_ts = raw_ts[:19]
            camera_name = sighting.get('camera_name', '')
            notes_by_sighting.append(f"<b>{display_ts} - {camera_name}:</b> {note}")

    if notes_by_sighting:
        notes_html = '<br/><br/>'.join(notes_by_sighting)
        elements.append(Paragraph(notes_html, styles['Normal']))
    else:
        elements.append(Paragraph('<i>No officer notes provided.</i>', styles['Italic']))

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
    Pull the CSV text, case_number, and officer_name out of the request.

    Supports multipart/form-data (file in 'csv' field + extra fields)
    and application/json ({'csv': '...', 'case_number': '...', ...}).

    Returns (csv_text, case_number, officer_name) – any may be None.
    """
    csv_text = None
    case_number = None
    officer_name = None

    if request_obj.files:
        csv_file = request_obj.files.get('csv')
        if csv_file:
            csv_text = csv_file.read().decode('utf-8')

    # Read extra FormData fields regardless of whether a file was uploaded
    if request_obj.form:
        case_number = request_obj.form.get('case_number') or None
        officer_name = request_obj.form.get('officer_name') or None

    if not csv_text and request_obj.is_json:
        data = request_obj.get_json(force=True, silent=True)
        if data:
            csv_text = data.get('csv')
            case_number = case_number or data.get('case_number')
            officer_name = officer_name or data.get('officer_name')

    return csv_text, case_number, officer_name


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
        csv_text, case_number, officer_name = _extract_csv_and_meta(request)

        if not csv_text:
            return jsonify({'error': 'CSV file or csv field is required'}), 400

        case_metadata, all_sightings = _parse_csv_data(csv_text, case_number, officer_name)

        # For preview include all sightings (include=true *or* not set)
        included = [s for s in all_sightings if s.get('include', True)]
        if not included:
            included = all_sightings  # fall back to all rows

        included.sort(key=lambda x: str(x.get('timestamp', '')))

        # Optional map (won't block PDF generation if it fails)
        camera_ids = list({s.get('camera_id') for s in included if s.get('camera_id')})
        camera_coords = _fetch_camera_coordinates(camera_ids) if camera_ids else {}
        map_image = _generate_mapbox_map(included, camera_coords) if camera_coords else None

        pdf_buffer = _create_court_ready_pdf(
            case_metadata, included, map_image, signature_info=None
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
        csv_text, case_number, officer_name = _extract_csv_and_meta(request)

        if not csv_text:
            return jsonify({'error': 'CSV file or csv field is required'}), 400

        case_metadata, all_sightings = _parse_csv_data(csv_text, case_number, officer_name)

        included_sightings = [s for s in all_sightings if s.get('include')]
        if not included_sightings:
            # Fall back to all sightings so the report is never empty
            included_sightings = all_sightings

        if not included_sightings:
            return jsonify({'error': 'CSV contains no sightings data'}), 400

        included_sightings.sort(key=lambda x: str(x.get('timestamp', '')))

        # Camera GPS coords + optional Mapbox map
        camera_ids = list({s.get('camera_id') for s in included_sightings if s.get('camera_id')})
        camera_coords = _fetch_camera_coordinates(camera_ids) if camera_ids else {}
        map_image = _generate_mapbox_map(included_sightings, camera_coords) if camera_coords else None

        # Digital signature
        signature_data = {
            'case_number': case_metadata.get('case_number'),
            'officer_name': case_metadata.get('officer_name'),
            'sighting_count': len(included_sightings),
        }
        signature_info = _generate_document_signature(signature_data)

        pdf_buffer = _create_court_ready_pdf(
            case_metadata, included_sightings, map_image, signature_info
        )

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
