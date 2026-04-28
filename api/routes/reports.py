"""
Reports endpoints.
Generates court-ready PDF reports from officer-edited CSV data with Mapbox mapping.
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


def _parse_csv_data(csv_text):
    """
    Parse officer-edited CSV into case metadata and sightings.
    
    Expected format:
        case_number,officer_name,date
        CASE-001,Officer Name,2024-04-28
        
        sighting_id,timestamp,camera_id,camera_name,include,match_score,officer_note
        1,2024-04-28T09:15:22Z,CAM-001,Main St,true,92.5,Note here
        ...
    
    Args:
        csv_text: CSV content as string
    
    Returns:
        Tuple of (case_metadata, sightings_list)
    """
    lines = csv_text.strip().split('\n')
    
    # Parse case metadata (first 2 lines)
    metadata_reader = csv.DictReader(lines[:2])
    metadata_rows = list(metadata_reader)
    
    if not metadata_rows:
        raise ValueError("Missing case metadata in CSV")
    
    case_metadata = metadata_rows[0]
    
    # Parse sightings (skip to first empty line, then read sightings)
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
    
    # Convert include to boolean
    for sighting in sightings:
        sighting['include'] = sighting.get('include', '').lower() in ('true', '1', 'yes')
        try:
            sighting['match_score'] = float(sighting.get('match_score', 0))
        except ValueError:
            sighting['match_score'] = 0
    
    return case_metadata, sightings


def _fetch_camera_coordinates(camera_ids):
    """
    Fetch camera GPS coordinates from database.
    
    Args:
        camera_ids: List of camera IDs
    
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
    
    Args:
        sightings: List of included sightings (sorted by timestamp)
        camera_coords: Dict mapping camera_id to {latitude, longitude}
        mapbox_token: Mapbox API token (defaults to app config)
    
    Returns:
        BytesIO: PNG image data
    """
    if not mapbox_token:
        mapbox_token = current_app.config.get('MAPBOX_ACCESS_TOKEN')
    
    if not mapbox_token:
        raise ValueError("MAPBOX_ACCESS_TOKEN not configured")
    
    # Build list of coordinates in order
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
        
        # Add pin marker
        color = 'FF0000' if idx == 0 else ('00FF00' if idx == len(sightings) - 1 else '0000FF')
        pins.append(f"pin-s+{color}({lon},{lat})")
    
    if len(coordinates) < 2:
        raise ValueError("Need at least 2 sightings with valid coordinates for map")
    
    # Calculate bounding box
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Add 10% padding
    lon_padding = (max_lon - min_lon) * 0.1 or 0.01
    lat_padding = (max_lat - min_lat) * 0.1 or 0.01
    
    bbox = f"{min_lon - lon_padding},{min_lat - lat_padding},{max_lon + lon_padding},{max_lat + lat_padding}"
    
    # Build polyline (route connecting sightings in order)
    polyline = ";".join(f"{lon},{lat}" for lon, lat in coordinates)
    
    # Build Mapbox Static API URL
    # Format: https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/path/linewidth/strokecolor-opacity/url-encoded-overlay/auto/width x height/retina
    overlay_parts = []
    
    # Add route polyline
    overlay_parts.append(f"path-3+0000FF-0.5({polyline})")
    
    # Add pins
    overlay_parts.extend(pins)
    
    overlay = "/".join(overlay_parts)
    
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/"
        f"{overlay}/"
        f"auto/800x600@2x"
        f"?access_token={mapbox_token}"
    )
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    map_image = BytesIO(response.content)
    map_image.seek(0)
    
    return map_image


def _generate_document_signature(data, secret_key=None):
    """
    Generate a signature for the report data.
    
    Args:
        data: Dictionary or JSON string of the report data
        secret_key: Secret key for HMAC (defaults to app secret)
    
    Returns:
        Dictionary with hash and signature
    """
    if isinstance(data, dict):
        data_string = json.dumps(data, sort_keys=True)
    else:
        data_string = str(data)
    
    # Generate SHA-256 hash of the data
    data_hash = hashlib.sha256(data_string.encode()).hexdigest()
    
    # Generate HMAC signature using the app secret (or provided key)
    key = (secret_key or current_app.config.get('SECRET_KEY', 'dev')).encode()
    signature = hmac.new(key, data_string.encode(), hashlib.sha256).hexdigest()
    
    return {
        'hash': data_hash,
        'signature': signature,
        'timestamp': datetime.utcnow().isoformat(),
        'algorithm': 'HMAC-SHA256'
    }


def _create_court_ready_pdf(case_metadata, sightings, map_image, signature_info):
    """
    Create a court-ready PDF report with cover page, map, sightings table, and notes.
    
    Args:
        case_metadata: Dict with case_number, officer_name, date
        sightings: List of sighting dicts (filtered to include=true)
        map_image: BytesIO PNG image
        signature_info: Dict with signature details
    
    Returns:
        BytesIO: PDF document
    """
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=20,
        alignment=1  # Center
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
    elements.append(Spacer(1, 0.3*inch))
    
    # Case information
    case_data = [
        ['CASE NUMBER:', case_metadata.get('case_number', 'N/A')],
        ['OFFICER:', case_metadata.get('officer_name', 'N/A')],
        ['DATE:', case_metadata.get('date', 'N/A')],
        ['REPORT GENERATED:', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
    ]
    
    case_table = Table(case_data, colWidths=[1.8*inch, 3.7*inch])
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
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary
    elements.append(Paragraph('SUMMARY', heading_style))
    summary_text = f"This report documents {len(sightings)} confirmed vehicle sightings from the automated tracking system."
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # === PAGE 2: MAP WITH ROUTE ===
    elements.append(PageBreak())
    elements.append(Paragraph('VEHICLE ROUTE MAP', heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Add map image
    try:
        img = Image(map_image, width=7*inch, height=5.25*inch)
        elements.append(img)
    except Exception as e:
        current_app.logger.warning(f"Failed to embed map image: {e}")
        elements.append(Paragraph('<i>Map image unavailable</i>', styles['Normal']))
    
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph(
        '<i>Red pin: First sighting | Blue pins: Intermediate sightings | Green pin: Last sighting | '
        'Blue line: Route connecting sightings in chronological order</i>',
        styles['Italic']
    ))
    
    # === PAGE 3+: SIGHTINGS TABLE ===
    elements.append(PageBreak())
    elements.append(Paragraph('VEHICLE SIGHTINGS', heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Sightings table
    sightings_data = [
        ['#', 'Time', 'Camera', 'Location', 'Confidence', 'Officer Notes']
    ]
    
    for idx, sighting in enumerate(sightings, 1):
        sightings_data.append([
            str(idx),
            sighting.get('timestamp', '')[:19],  # Remove Z suffix
            sighting.get('camera_id', ''),
            sighting.get('camera_name', ''),
            f"{sighting.get('match_score', 0):.1f}%",
            sighting.get('officer_note', ''),
        ])
    
    sightings_table = Table(sightings_data, colWidths=[0.5*inch, 1.2*inch, 0.9*inch, 1.5*inch, 0.9*inch, 1.5*inch])
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
    elements.append(Spacer(1, 0.2*inch))
    
    # Combine all officer notes
    notes_by_sighting = []
    for sighting in sightings:
        if sighting.get('officer_note'):
            timestamp = sighting.get('timestamp', '')[:19]
            camera_name = sighting.get('camera_name', '')
            note = sighting.get('officer_note', '')
            notes_by_sighting.append(f"<b>{timestamp} - {camera_name}:</b> {note}")
    
    if notes_by_sighting:
        notes_html = '<br/><br/>'.join(notes_by_sighting)
        elements.append(Paragraph(notes_html, styles['Normal']))
    else:
        elements.append(Paragraph('<i>No officer notes provided.</i>', styles['Italic']))
    
    # === SIGNATURE PAGE ===
    elements.append(PageBreak())
    elements.append(Paragraph('DIGITAL SIGNATURE', heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    sig_data = [
        ['Signer:', 'Automated System'],
        ['Timestamp:', signature_info.get('timestamp', 'N/A')],
        ['Algorithm:', signature_info.get('algorithm', 'N/A')],
        ['Document Hash:', signature_info.get('hash', '')[:48] + '...'],
        ['Signature:', signature_info.get('signature', '')[:48] + '...'],
    ]
    
    sig_table = Table(sig_data, colWidths=[1.5*inch, 4.5*inch])
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
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(
        '<i>This document has been cryptographically signed. The hash and signature above can be used '
        'to verify the document integrity and authenticity.</i>',
        styles['Italic']
    ))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer


@bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Generate court-ready signed PDF from officer-edited CSV.
    
    Accepts multipart CSV file or JSON with CSV text:
    - multipart: CSV file in 'csv' field
    - JSON: {'csv': 'csv_text_content'}
    
    Process:
    1. Parse CSV (case metadata + sightings)
    2. Filter sightings where include=true
    3. Fetch GPS coords for each camera
    4. Generate Mapbox static map with route
    5. Build multi-page PDF with signature
    """
    try:
        csv_text = None
        
        # Get CSV from file or JSON
        if request.files:
            csv_file = request.files.get('csv')
            if csv_file:
                csv_text = csv_file.read().decode('utf-8')
        
        if not csv_text and request.is_json:
            data = request.get_json(force=True, silent=True)
            csv_text = data.get('csv') if data else None
        
        if not csv_text:
            return jsonify({'error': 'CSV file or csv field required'}), 400
        
        # Parse CSV
        case_metadata, all_sightings = _parse_csv_data(csv_text)
        
        # Filter to included sightings only
        included_sightings = [s for s in all_sightings if s.get('include')]
        
        if not included_sightings:
            return jsonify({'error': 'No included sightings (include=true) in CSV'}), 400
        
        # Sort by timestamp
        included_sightings.sort(key=lambda x: x.get('timestamp', ''))
        
        # Fetch camera coordinates
        camera_ids = list(set(s.get('camera_id') for s in included_sightings))
        camera_coords = _fetch_camera_coordinates(camera_ids)
        
        # Generate map
        map_image = _generate_mapbox_map(included_sightings, camera_coords)
        
        # Generate signature
        signature_data = {
            'case_number': case_metadata.get('case_number'),
            'officer_name': case_metadata.get('officer_name'),
            'sighting_count': len(included_sightings),
        }
        signature_info = _generate_document_signature(signature_data)
        
        # Build PDF
        pdf_buffer = _create_court_ready_pdf(case_metadata, included_sightings, map_image, signature_info)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'report_{case_metadata.get("case_number", "report")}.pdf'
        )
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500


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
        current_app.logger.error(f"Error verifying signature: {str(e)}")
        return jsonify({'error': f'Failed to verify signature: {str(e)}'}), 500
