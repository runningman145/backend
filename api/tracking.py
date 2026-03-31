import os
from datetime import datetime
from io import BytesIO
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from .db import get_db

bp = Blueprint('tracking', __name__, url_prefix='/tracking')

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_upload_folder():
    """Get or create upload folder."""
    upload_folder = os.path.join(current_app.instance_path, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    return upload_folder


@bp.route('/detections', methods=['GET'])
def get_detections():
    """Return all detections with the capturing camera's coordinates."""
    db = get_db()
    rows = db.execute(
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
        ORDER BY d.captured_at DESC
        '''
    ).fetchall()

    return jsonify([
        {
            'detection_id': row['id'],
            'captured_at': row['captured_at'].isoformat(),
            'camera': {
                'id': row['camera_id'],
                'name': row['camera_name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
            },
        }
        for row in rows
    ])


@bp.route('/detections', methods=['POST'])
def add_detection():
    """Record a new car detection submitted by the ML model."""
    data = request.get_json()

    if not data or 'camera_id' not in data:
        return jsonify({'error': 'camera_id is required'}), 400

    camera_id = data['camera_id']
    db = get_db()

    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()

    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    cursor = db.execute(
        'INSERT INTO detections (camera_id) VALUES (?)', (camera_id,)
    )
    db.commit()

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
        'captured_at': new_row['captured_at'].isoformat(),
        'camera': {
            'id': new_row['camera_id'],
            'name': new_row['camera_name'],
            'latitude': new_row['latitude'],
            'longitude': new_row['longitude'],
        },
    }), 201


@bp.route('/upload', methods=['POST'])
def upload_media():
    """Upload picture or video file for a detection."""
    # Check if request has file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    camera_id = request.form.get('camera_id')
    detection_id = request.form.get('detection_id')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return jsonify({'error': f'File type not allowed. Allowed: {allowed}'}), 400
    
    # Verify camera exists
    db = get_db()
    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()
    
    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404
    
    # If detection_id provided, verify it exists
    if detection_id:
        detection = db.execute(
            'SELECT id FROM detections WHERE id = ? AND camera_id = ?',
            (detection_id, camera_id)
        ).fetchone()
        
        if detection is None:
            return jsonify({'error': f'Detection {detection_id} not found for camera {camera_id}'}), 404
    
    # Save file securely
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    
    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, filename)
    
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    return jsonify({
        'message': 'File uploaded successfully',
        'filename': filename,
        'camera_id': camera_id,
        'detection_id': detection_id,
        'upload_path': f'/uploads/{filename}'
    }), 201


@bp.route('/uploads/<filename>', methods=['GET'])
def download_media(filename):
    """Download uploaded media file."""
    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, secure_filename(filename))
    
    # Security: check file exists and is in upload folder
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return jsonify({'error': f'File {filename} not found'}), 404
    
    # Verify the file is actually in the upload folder
    if not os.path.abspath(filepath).startswith(os.path.abspath(upload_folder)):
        return jsonify({'error': 'Invalid file path'}), 403
    
    return jsonify({
        'filename': filename,
        'path': filepath,
        'size': os.path.getsize(filepath)
    })


@bp.route('/detections/<int:detection_id>/export-pdf', methods=['GET'])
def export_detection_pdf(detection_id):
    """Export a detection to PDF format."""
    db = get_db()
    
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
    
    # Create PDF in memory
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
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
    detection_data = [
        ['Detection ID:', str(detection['id'])],
        ['Captured At:', detection['captured_at'].isoformat()],
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
        ['Camera ID:', detection['camera_id']],
        ['Camera Name:', detection['camera_name']],
        ['Status:', detection['status']],
        ['Latitude:', detection['latitude']],
        ['Longitude:', detection['longitude']],
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
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'detection_{detection_id}.pdf'
    )
