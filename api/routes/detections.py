"""
Detection endpoints.
Handles creation and retrieval of vehicle detections, and PDF export.
"""
import base64
import json
import os
from io import BytesIO
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file, current_app
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage
from ..db import get_db

bp = Blueprint('detections', __name__, url_prefix='/detections')


@bp.route('', methods=['GET'])
def get_detections():
    """Return all detections with the capturing camera's coordinates."""
    db = get_db()
    
    query = '''
        SELECT
            d.id,
            d.captured_at,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        ORDER BY d.id DESC
    '''
    
    rows = db.execute(query).fetchall()
    
    return jsonify([
        {
            'detection_id': row['id'],
            'captured_at': row['captured_at'].isoformat() if hasattr(row['captured_at'], 'isoformat') else str(row['captured_at']),
            'camera': {
                'id': row['camera_id'],
                'name': row['camera_name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
            },
        }
        for row in rows
    ])


@bp.route('/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get a specific detection by ID."""
    db = get_db()
    
    detection = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            d.camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude,
            c.status
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.id = ?
        ''',
        (detection_id,)
    ).fetchone()
    
    if detection is None:
        return jsonify({'error': f'Detection {detection_id} not found'}), 404
    
    return jsonify({
        'detection_id': detection['id'],
        'captured_at': detection['captured_at'].isoformat() if hasattr(detection['captured_at'], 'isoformat') else str(detection['captured_at']),
        'camera': {
            'id': detection['camera_id'],
            'name': detection['camera_name'],
            'latitude': detection['latitude'],
            'longitude': detection['longitude'],
            'status': detection['status']
        }
    })


@bp.route('', methods=['POST'])
def add_detection():
    """Record a new car detection submitted by the ML model."""
    data = request.get_json(force=True, silent=True)

    if not data or 'camera_id' not in data:
        return jsonify({'error': 'camera_id is required'}), 400

    camera_id = data['camera_id']
    
    db = get_db()

    # Verify camera exists
    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()

    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    # Insert detection
    cursor = db.execute(
        '''INSERT INTO detections (camera_id) 
           VALUES (?)''',
        (camera_id,)
    )
    db.commit()

    # Fetch the created detection
    new_row = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.id = ?
        ''',
        (cursor.lastrowid,)
    ).fetchone()

    return jsonify({
        'detection_id': new_row['id'],
        'captured_at': new_row['captured_at'].isoformat() if hasattr(new_row['captured_at'], 'isoformat') else str(new_row['captured_at']),
        'camera': {
            'id': new_row['camera_id'],
            'name': new_row['camera_name'],
            'latitude': new_row['latitude'],
            'longitude': new_row['longitude'],
        },
    }), 201




@bp.route('/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """Delete a detection and its associated vehicle detections."""
    db = get_db()
    
    # Check if detection exists
    detection = db.execute(
        'SELECT id FROM detections WHERE id = ?',
        (detection_id,)
    ).fetchone()
    
    if detection is None:
        return jsonify({'error': f'Detection {detection_id} not found'}), 404
    
    # Delete associated vehicle detections first (foreign key constraint)
    db.execute('DELETE FROM vehicle_detections WHERE detection_id = ?', (detection_id,))
    
    # Delete detection
    db.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
    db.commit()
    
    return jsonify({'message': 'Detection deleted successfully'}), 200



@bp.route('/<int:detection_id>/export-pdf', methods=['GET'])
def export_detection_pdf(detection_id):
    """Export a detection to PDF format with enhanced details including uploaded and matched images."""
    db = get_db()
    
    # Fetch detection with all details
    detection = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude,
            c.status
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.id = ?
        ''',
        (detection_id,)
    ).fetchone()
    
    if detection is None:
        return jsonify({'error': f'Detection {detection_id} not found'}), 404
    
    # Fetch query image filename (uploaded image) from jobs
    query_job = db.execute(
        '''
        SELECT query_image_filename FROM jobs 
        WHERE detection_id = ? 
        LIMIT 1
        ''',
        (detection_id,)
    ).fetchone()
    
    query_image_filename = query_job['query_image_filename'] if query_job else None
    
    # Fetch vehicle detections (matched vehicles from this query)
    vehicle_detections = db.execute(
        '''
        SELECT 
            vd.id,
            vd.timestamp,
            vd.match_score,
            vd.camera_id,
            vd.box_x1,
            vd.box_y1,
            vd.box_x2,
            vd.box_y2,
            c.name as camera_name
        FROM vehicle_detections vd
        JOIN cameras c ON vd.camera_id = c.id
        WHERE vd.detection_id = ?
        ORDER BY vd.match_score DESC
        ''',
        (detection_id,)
    ).fetchall()
    
    # Create PDF in memory
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2d5aa6'),
        spaceAfter=12,
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#404040'),
        spaceAfter=8,
    )
    
    # Title
    elements.append(Paragraph('Vehicle Detection Report', title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Detection Info Section
    elements.append(Paragraph('Detection Information', heading_style))
    captured_at_str = detection['captured_at'].isoformat() if hasattr(detection['captured_at'], 'isoformat') else str(detection['captured_at'])
    
    detection_data = [
        ['Detection ID:', str(detection['id'])],
        ['Captured At:', captured_at_str],
    ]
    
    detection_table = Table(detection_data, colWidths=[2 * inch, 3.5 * inch])
    detection_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(detection_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Camera Info Section
    elements.append(Paragraph('Camera Information', heading_style))
    camera_data = [
        ['Camera ID:', str(detection['camera_id'])],
        ['Camera Name:', detection['camera_name']],
        ['Status:', detection['status']],
        ['Latitude:', f"{float(detection['latitude']):.6f}" if detection['latitude'] else 'N/A'],
        ['Longitude:', f"{float(detection['longitude']):.6f}" if detection['longitude'] else 'N/A'],
    ]
    
    camera_table = Table(camera_data, colWidths=[2 * inch, 3.5 * inch])
    camera_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(camera_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Uploaded/Query Image Section
    if query_image_filename:
        elements.append(Paragraph('Uploaded Query Image', heading_style))
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        query_image_path = os.path.join(upload_folder, query_image_filename)
        
        if os.path.exists(query_image_path):
            try:
                # Get image dimensions to scale appropriately
                pil_image = PILImage.open(query_image_path)
                img_width, img_height = pil_image.size
                
                # Scale to fit page width (max 5 inches)
                max_width = 5 * inch
                scale_factor = max_width / (img_width / 96)  # Assuming 96 DPI
                if img_height / 96 * scale_factor > 4 * inch:
                    scale_factor = (4 * inch) / (img_height / 96)
                
                display_width = (img_width / 96) * scale_factor
                display_height = (img_height / 96) * scale_factor
                
                img = Image(query_image_path, width=display_width, height=display_height)
                elements.append(img)
                elements.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                elements.append(Paragraph(f'Could not load query image: {str(e)}', styles['Normal']))
                elements.append(Spacer(1, 0.2 * inch))
        else:
            elements.append(Paragraph(f'Query image not found: {query_image_filename}', styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))
    
    # -----------------------------------------------------------------------
    # Pull frame images stored in the job's result_data JSON
    # -----------------------------------------------------------------------
    job_frame_matches = []
    job_result_row = db.execute(
        'SELECT result_data FROM jobs WHERE detection_id = ? ORDER BY created_at DESC LIMIT 1',
        (detection_id,)
    ).fetchone()
    if job_result_row and job_result_row['result_data']:
        try:
            job_result = json.loads(job_result_row['result_data'])
            job_frame_matches = [
                m for m in job_result.get('matches', [])
                if m.get('frame_image')
            ]
        except Exception as e:
            current_app.logger.warning(f'Could not parse job result_data for frames: {e}')

    # Vehicle Detections (Matched Vehicles) Section
    if vehicle_detections:
        elements.append(Paragraph('Matched Vehicle Detections', heading_style))
        
        # Summary table
        vd_summary_data = [['Det ID', 'Camera', 'Timestamp (s)', 'Match Score', 'Box Coordinates']]
        for vd in vehicle_detections:
            box_coords = f"({vd['box_x1']}, {vd['box_y1']}, {vd['box_x2']}, {vd['box_y2']})"
            vd_summary_data.append([
                str(vd['id']),
                vd['camera_name'],
                f"{vd['timestamp']:.2f}",
                f"{vd['match_score']:.2f}%" if vd.get('match_score') else 'N/A',
                box_coords
            ])
        
        vd_table = Table(vd_summary_data, colWidths=[0.8 * inch, 1.5 * inch, 1.2 * inch, 1.2 * inch, 1.5 * inch])
        vd_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(vd_table)
        elements.append(Spacer(1, 0.2 * inch))

    # -----------------------------------------------------------------------
    # Vehicle Frame Images Section
    # Embed each matched-vehicle crop (base64 JPEG) stored in the job results.
    # -----------------------------------------------------------------------
    if job_frame_matches:
        elements.append(PageBreak())
        elements.append(Paragraph('Matched Vehicle Frames', heading_style))
        elements.append(Spacer(1, 0.1 * inch))

        for idx, match in enumerate(job_frame_matches, 1):
            try:
                # Strip data-URI prefix if present
                b64_data = match['frame_image']
                if ',' in b64_data:
                    b64_data = b64_data.split(',', 1)[1]

                img_bytes = base64.b64decode(b64_data)
                img_buffer = BytesIO(img_bytes)

                # Determine display dimensions (max 3 in wide, proportional)
                pil_img = PILImage.open(BytesIO(img_bytes))
                iw, ih = pil_img.size
                max_w = 3.0 * inch
                scale = max_w / iw if iw > 0 else 1
                disp_w = iw * scale
                disp_h = ih * scale
                # Cap height at 2.5 inches
                if disp_h > 2.5 * inch:
                    scale = (2.5 * inch) / ih
                    disp_w = iw * scale
                    disp_h = ih * scale

                # Sub-caption: timestamp + score
                ts = match.get('time', 'N/A')
                score = match.get('match_percent', 'N/A')
                caption = (
                    f'Frame {idx} — Timestamp: {ts}s | '
                    f'Match: {score}%'
                )
                elements.append(Paragraph(caption, subheading_style))

                rl_img = Image(img_buffer, width=disp_w, height=disp_h)
                elements.append(rl_img)
                elements.append(Spacer(1, 0.25 * inch))

            except Exception as e:
                current_app.logger.warning(
                    f'Could not embed frame image for match {idx}: {e}'
                )
                elements.append(
                    Paragraph(f'Frame {idx}: image unavailable ({e})', styles['Normal'])
                )
                elements.append(Spacer(1, 0.1 * inch))
    
    # Add footer with generation timestamp
    elements.append(Spacer(1, 0.5 * inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1  # Center alignment
    )
    elements.append(Paragraph(
        f'Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'detection_{detection_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )


@bp.route('/stats', methods=['GET'])
def detection_stats():
    """Get statistics about detections."""
    db = get_db()
    
    # Get total count
    total = db.execute('SELECT COUNT(*) as count FROM detections').fetchone()['count']
    
    # Get detections by camera
    by_camera = db.execute('''
        SELECT 
            c.name as camera_name,
            COUNT(d.id) as detection_count
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        GROUP BY d.camera_id
        ORDER BY detection_count DESC
    ''').fetchall()
    
    # Get detections over time (last 30 days)
    by_date = db.execute('''
        SELECT 
            DATE(d.captured_at) as date,
            COUNT(*) as count
        FROM detections d
        WHERE d.captured_at >= DATE('now', '-30 days')
        GROUP BY DATE(d.captured_at)
        ORDER BY date DESC
    ''').fetchall()
    
    # Get average match score from vehicle_detections
    avg_score = db.execute('''
        SELECT AVG(match_score) as avg_score
        FROM vehicle_detections
        WHERE match_score IS NOT NULL
    ''').fetchone()['avg_score']

    # Get match confidence distribution from vehicle_detections
    confidence_dist_rows = db.execute('''
        SELECT
            CASE
                WHEN match_score >= 90 THEN '90-100%'
                WHEN match_score >= 80 THEN '80-90%'
                WHEN match_score >= 70 THEN '70-80%'
                ELSE 'Below 70%'
            END as score_range,
            COUNT(*) as count
        FROM vehicle_detections
        WHERE match_score IS NOT NULL
        GROUP BY score_range
    ''').fetchall()

    # Build ordered distribution list
    dist_map = {row['score_range']: row['count'] for row in confidence_dist_rows}
    confidence_distribution = [
        {'label': '90-100%', 'value': dist_map.get('90-100%', 0)},
        {'label': '80-90%',  'value': dist_map.get('80-90%', 0)},
        {'label': '70-80%',  'value': dist_map.get('70-80%', 0)},
        {'label': 'Below 70%', 'value': dist_map.get('Below 70%', 0)},
    ]

    return jsonify({
        'total_detections': total,
        'average_match_score': round(avg_score, 2) if avg_score else None,
        'detections_by_camera': [
            {'camera_name': row['camera_name'], 'count': row['detection_count']}
            for row in by_camera
        ],
        'detections_by_date': [
            {'date': row['date'], 'count': row['count']}
            for row in by_date
        ],
        'confidence_distribution': confidence_distribution,
    })