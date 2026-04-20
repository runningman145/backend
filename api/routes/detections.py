"""
Detection endpoints.
Handles creation and retrieval of vehicle detections, and PDF export.
"""
import os
from io import BytesIO
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file, current_app
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from ..db import get_db

bp = Blueprint('detections', __name__, url_prefix='/detections')


@bp.route('', methods=['GET'])
def get_detections():
    """Return all detections with the capturing camera's coordinates."""
    db = get_db()
    
    # Add pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    camera_id = request.args.get('camera_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Build query with filters
    query = '''
        SELECT
            d.id,
            d.captured_at,
            d.match_score,
            d.track_id,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE 1=1
    '''
    params = []
    
    if camera_id:
        query += ' AND d.camera_id = ?'
        params.append(camera_id)
    
    if start_date:
        query += ' AND d.captured_at >= ?'
        params.append(start_date)
    
    if end_date:
        query += ' AND d.captured_at <= ?'
        params.append(end_date)
    
    # Add ordering and pagination
    query += ' ORDER BY d.captured_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])
    
    rows = db.execute(query, params).fetchall()
    
    # Get total count for pagination
    count_query = '''
        SELECT COUNT(*) as total
        FROM detections d
        WHERE 1=1
    '''
    count_params = []
    if camera_id:
        count_query += ' AND d.camera_id = ?'
        count_params.append(camera_id)
    if start_date:
        count_query += ' AND d.captured_at >= ?'
        count_params.append(start_date)
    if end_date:
        count_query += ' AND d.captured_at <= ?'
        count_params.append(end_date)
    
    total = db.execute(count_query, count_params).fetchone()['total']
    
    return jsonify({
        'detections': [
            {
                'detection_id': row['id'],
                'captured_at': row['captured_at'].isoformat() if hasattr(row['captured_at'], 'isoformat') else str(row['captured_at']),
                'match_score': row.get('match_score'),
                'track_id': row.get('track_id'),
                'camera': {
                    'id': row['camera_id'],
                    'name': row['camera_name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                },
            }
            for row in rows
        ],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page if per_page > 0 else 0
        }
    })


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
            d.query_embedding,
            d.match_score,
            d.track_id,
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
    
    # Get associated matches if any
    matches = db.execute(
        '''
        SELECT 
            vm.id,
            vm.timestamp,
            vm.match_score,
            vm.track_id,
            vm.camera_id,
            c.name as camera_name
        FROM vehicle_matches vm
        JOIN cameras c ON vm.camera_id = c.id
        WHERE vm.detection_id = ?
        ORDER BY vm.timestamp
        ''',
        (detection_id,)
    ).fetchall()
    
    return jsonify({
        'detection_id': detection['id'],
        'captured_at': detection['captured_at'].isoformat() if hasattr(detection['captured_at'], 'isoformat') else str(detection['captured_at']),
        'camera': {
            'id': detection['camera_id'],
            'name': detection['camera_name'],
            'latitude': detection['latitude'],
            'longitude': detection['longitude'],
            'status': detection['status']
        },
        'query_embedding': detection['query_embedding'] if detection['query_embedding'] else None,
        'match_score': detection.get('match_score'),
        'track_id': detection.get('track_id'),
        'matches': [
            {
                'id': m['id'],
                'timestamp': m['timestamp'],
                'match_score': m['match_score'],
                'track_id': m['track_id'],
                'camera_id': m['camera_id'],
                'camera_name': m['camera_name']
            }
            for m in matches
        ]
    })


@bp.route('', methods=['POST'])
def add_detection():
    """Record a new car detection submitted by the ML model."""
    data = request.get_json(force=True, silent=True)

    if not data or 'camera_id' not in data:
        return jsonify({'error': 'camera_id is required'}), 400

    camera_id = data['camera_id']
    query_embedding = data.get('query_embedding')  # Optional embedding
    match_score = data.get('match_score')
    track_id = data.get('track_id')
    
    db = get_db()

    # Verify camera exists
    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()

    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    # Insert detection
    cursor = db.execute(
        '''INSERT INTO detections (camera_id, query_embedding, match_score, track_id) 
           VALUES (?, ?, ?, ?)''',
        (camera_id, query_embedding, match_score, track_id)
    )
    db.commit()

    # Fetch the created detection
    new_row = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            d.match_score,
            d.track_id,
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
        'match_score': new_row.get('match_score'),
        'track_id': new_row.get('track_id'),
        'camera': {
            'id': new_row['camera_id'],
            'name': new_row['camera_name'],
            'latitude': new_row['latitude'],
            'longitude': new_row['longitude'],
        },
    }), 201


@bp.route('/<int:detection_id>', methods=['PUT'])
def update_detection(detection_id):
    """Update a detection (e.g., add track_id or match_score)."""
    data = request.get_json(force=True, silent=True)
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    db = get_db()
    
    # Check if detection exists
    detection = db.execute(
        'SELECT id FROM detections WHERE id = ?',
        (detection_id,)
    ).fetchone()
    
    if detection is None:
        return jsonify({'error': f'Detection {detection_id} not found'}), 404
    
    # Build update query dynamically
    update_fields = []
    params = []
    
    if 'track_id' in data:
        update_fields.append('track_id = ?')
        params.append(data['track_id'])
    
    if 'match_score' in data:
        update_fields.append('match_score = ?')
        params.append(data['match_score'])
    
    if 'query_embedding' in data:
        update_fields.append('query_embedding = ?')
        params.append(data['query_embedding'])
    
    if not update_fields:
        return jsonify({'error': 'No valid fields to update'}), 400
    
    params.append(detection_id)
    query = f'UPDATE detections SET {", ".join(update_fields)} WHERE id = ?'
    
    db.execute(query, params)
    db.commit()
    
    return jsonify({'message': 'Detection updated successfully'}), 200


@bp.route('/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """Delete a detection and its associated matches."""
    db = get_db()
    
    # Check if detection exists
    detection = db.execute(
        'SELECT id FROM detections WHERE id = ?',
        (detection_id,)
    ).fetchone()
    
    if detection is None:
        return jsonify({'error': f'Detection {detection_id} not found'}), 404
    
    # Delete associated matches first (foreign key constraint)
    db.execute('DELETE FROM vehicle_matches WHERE detection_id = ?', (detection_id,))
    
    # Delete detection
    db.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
    db.commit()
    
    return jsonify({'message': 'Detection deleted successfully'}), 200


@bp.route('/<int:detection_id>/export-pdf', methods=['GET'])
def export_detection_pdf(detection_id):
    """Export a detection to PDF format with enhanced details."""
    db = get_db()
    
    # Fetch detection with all details
    detection = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            d.match_score,
            d.track_id,
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
    
    # Fetch associated matches
    matches = db.execute(
        '''
        SELECT 
            vm.id,
            vm.timestamp,
            vm.match_score,
            vm.track_id,
            c.name as camera_name
        FROM vehicle_matches vm
        JOIN cameras c ON vm.camera_id = c.id
        WHERE vm.detection_id = ?
        ORDER BY vm.timestamp
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
    
    # Title
    elements.append(Paragraph('Vehicle Detection Report', title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Detection Info Section
    elements.append(Paragraph('Detection Information', heading_style))
    captured_at_str = detection['captured_at'].isoformat() if hasattr(detection['captured_at'], 'isoformat') else str(detection['captured_at'])
    
    detection_data = [
        ['Detection ID:', str(detection['id'])],
        ['Captured At:', captured_at_str],
        ['Match Score:', f"{detection['match_score']:.2f}%" if detection.get('match_score') else 'N/A'],
        ['Track ID:', str(detection['track_id']) if detection.get('track_id') else 'N/A'],
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
        ['Latitude:', f"{detection['latitude']:.6f}" if detection['latitude'] else 'N/A'],
        ['Longitude:', f"{detection['longitude']:.6f}" if detection['longitude'] else 'N/A'],
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
    
    # Add matches section if there are any
    if matches:
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph('Vehicle Matches', heading_style))
        
        # Prepare match data for table
        match_data = [['Match ID', 'Timestamp', 'Match Score', 'Track ID', 'Camera Name']]
        for match in matches:
            match_data.append([
                str(match['id']),
                match['timestamp'][:19] if match['timestamp'] else 'N/A',
                f"{match['match_score']:.2f}%" if match.get('match_score') else 'N/A',
                str(match['track_id']) if match.get('track_id') else 'N/A',
                match['camera_name']
            ])
        
        match_table = Table(match_data, colWidths=[1.2 * inch, 1.8 * inch, 1.2 * inch, 1.2 * inch, 1.6 * inch])
        match_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(match_table)
    
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
    
    # Get average match score
    avg_score = db.execute('''
        SELECT AVG(match_score) as avg_score 
        FROM detections 
        WHERE match_score IS NOT NULL
    ''').fetchone()['avg_score']
    
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
        ]
    })