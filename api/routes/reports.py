"""
Reports endpoints.
Provides endpoints for generating PDF reports from job results with preview and signing capabilities.
"""
import json
import hashlib
import hmac
from datetime import datetime
from io import BytesIO
from flask import Blueprint, jsonify, request, send_file, current_app
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

bp = Blueprint('reports', __name__, url_prefix='/reports')


def _create_pdf_from_data(report_data, include_signature=False, signature_info=None):
    """
    Create a PDF document from report data.
    
    Args:
        report_data: Dictionary containing report data to render
        include_signature: Whether to include signature section
        signature_info: Dictionary with signature details (signer_name, timestamp, hash)
    
    Returns:
        BytesIO: PDF document in memory
    """
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
    report_title = report_data.get('title', 'Detection Results Report')
    elements.append(Paragraph(report_title, title_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Metadata section
    if 'metadata' in report_data:
        elements.append(Paragraph('Metadata', heading_style))
        metadata = report_data['metadata']
        
        metadata_rows = [[str(k), str(v)] for k, v in metadata.items()]
        metadata_table = Table(metadata_rows, colWidths=[2 * inch, 3.5 * inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.2 * inch))
    
    # Results section
    if 'results' in report_data:
        elements.append(Paragraph('Results', heading_style))
        results = report_data['results']
        
        if isinstance(results, list):
            # List of detections
            if results:
                # Extract all unique keys for column headers
                all_keys = set()
                for item in results:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                
                all_keys = sorted(list(all_keys))
                
                # Build table
                table_data = [all_keys]  # Header row
                for item in results:
                    if isinstance(item, dict):
                        row = [str(item.get(key, '')) for key in all_keys]
                    else:
                        row = [str(item)]
                    table_data.append(row)
                
                # Limit to reasonable page size
                if len(table_data) > 50:
                    # Add pagination indicator
                    col_width = 5.5 * inch / len(all_keys) if all_keys else 1 * inch
                    results_table = Table(table_data[:50], colWidths=[col_width] * len(all_keys))
                else:
                    col_width = 5.5 * inch / len(all_keys) if all_keys else 1 * inch
                    results_table = Table(table_data, colWidths=[col_width] * len(all_keys))
                
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ]))
                elements.append(results_table)
                
                if len(table_data) > 50:
                    elements.append(Spacer(1, 0.1 * inch))
                    elements.append(Paragraph(
                        f'<i>Showing first 50 of {len(table_data) - 1} results. '
                        'Download full CSV for complete data.</i>',
                        styles['Normal']
                    ))
        else:
            # Single result object
            result_rows = [[str(k), str(v)] for k, v in results.items()]
            results_table = Table(result_rows, colWidths=[2 * inch, 3.5 * inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(results_table)
        
        elements.append(Spacer(1, 0.3 * inch))
    
    # Signature section
    if include_signature and signature_info:
        elements.append(PageBreak())
        elements.append(Paragraph('Document Signature', heading_style))
        
        sig_data = [
            ['Signer:', signature_info.get('signer_name', 'N/A')],
            ['Timestamp:', signature_info.get('timestamp', 'N/A')],
            ['Document Hash:', signature_info.get('hash', '')[:64] + '...'],
            ['Signature:', signature_info.get('signature', '')[:64] + '...'],
        ]
        
        sig_table = Table(sig_data, colWidths=[1.5 * inch, 4 * inch])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(sig_table)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            '<i>This document has been digitally signed and its integrity is verified '
            'through the hash and signature provided above.</i>',
            styles['Normal']
        ))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer


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


@bp.route('/preview', methods=['POST'])
def preview_report():
    """
    Generate a PDF preview of a report.
    
    Request body (JSON):
    {
        'title': 'Report Title',
        'metadata': {
            'job_id': '...',
            'created_at': '...',
            ...
        },
        'results': [
            {...},
            {...}
        ] or {...}
    }
    
    Returns:
        PDF file (unsigned preview)
    """
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        # Validate required fields
        if 'results' not in data:
            return jsonify({'error': 'results field is required'}), 400
        
        # Generate PDF without signature
        pdf_buffer = _create_pdf_from_data(data, include_signature=False)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='report_preview.pdf'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error generating PDF preview: {str(e)}")
        return jsonify({'error': f'Failed to generate preview: {str(e)}'}), 500


@bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Generate a signed PDF report.
    
    Request body (JSON):
    {
        'title': 'Report Title',
        'metadata': {
            'job_id': '...',
            'created_at': '...',
            ...
        },
        'results': [
            {...},
            {...}
        ] or {...},
        'signer_name': 'System' (optional, default: 'System')
    }
    
    Returns:
        Signed PDF file with embedded signature and hash
    """
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        # Validate required fields
        if 'results' not in data:
            return jsonify({'error': 'results field is required'}), 400
        
        # Generate signature for the data
        signature_info = _generate_document_signature(data)
        
        # Add signer name
        signer_name = data.get('signer_name', 'System')
        signature_info['signer_name'] = signer_name
        
        # Generate PDF with signature
        pdf_buffer = _create_pdf_from_data(
            data,
            include_signature=True,
            signature_info=signature_info
        )
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='report_signed.pdf'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error generating signed PDF: {str(e)}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500


@bp.route('/verify-signature', methods=['POST'])
def verify_signature():
    """
    Verify a document signature (for external validation).
    
    Request body (JSON):
    {
        'data': {...},  // original data object
        'signature': '...',  // the signature to verify
        'hash': '...'  // the document hash
    }
    
    Returns:
        JSON with verification result
    """
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'data' not in data or 'signature' not in data:
            return jsonify({'error': 'data and signature fields are required'}), 400
        
        # Regenerate signature
        generated_sig = _generate_document_signature(data['data'])
        
        # Compare signatures
        is_valid = (
            generated_sig['signature'] == data['signature'] and
            generated_sig['hash'] == data.get('hash', '')
        )
        
        return jsonify({
            'valid': is_valid,
            'algorithm': 'HMAC-SHA256',
            'message': 'Signature valid' if is_valid else 'Signature invalid or data has been modified'
        })
    
    except Exception as e:
        current_app.logger.error(f"Error verifying signature: {str(e)}")
        return jsonify({'error': f'Failed to verify signature: {str(e)}'}), 500
