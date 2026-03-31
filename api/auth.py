import functools
import uuid
from datetime import datetime, timedelta
from flask import (
    Blueprint, flash, g, request, jsonify, current_app
)

from werkzeug.security import check_password_hash, generate_password_hash
import jwt
from api.db import get_db

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=['POST'])
def register():
    """ Admin should register new users"""
    user = request.get_json()
    if not user:
        return jsonify({'error': 'User information required'}), 400
    
    # Validate required fields
    username = user.get('username')
    password = user.get('password')
    email = user.get('email')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    if not password:
        return jsonify({'error': 'Password is required'}), 400
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    error = None
    user_id = str(uuid.uuid4())
    db = get_db()
    
    try:
        db.execute(
            'INSERT INTO users (id, email, username, password) VALUES (?, ?, ?, ?)',
            (user_id, email, username, generate_password_hash(password))
        )
        db.commit()
    except db.IntegrityError:
        error = f"User {username} is already registered."
        return jsonify({'error': error}), 409
    
    return jsonify({'message': 'User registered successfully', 'user_id': user_id}), 201

@bp.route('/login', methods=['POST'])
def login():
    """ A user can login to view their dashboard and make queries"""
    user = request.get_json()
    if not user:
        return jsonify({'error': 'Bad request'}), 400
    
    email = user.get('email')
    password = user.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    db = get_db()
    db_user = db.execute(
        'SELECT * FROM users WHERE email = ?', (email,)
    ).fetchone()
    
    if db_user is None:
        return jsonify({'error': 'User not found'}), 404
    
    if not check_password_hash(db_user['password'], password):
        return jsonify({'error': 'Invalid password'}), 401
    
    # Generate JWT token with 24-hour expiration
    token = jwt.encode(
        {
            'user_id': db_user['id'],
            'email': db_user['email'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        },
        current_app.config.get('SECRET_KEY', 'dev'),
        algorithm='HS256'
    )
    return jsonify({'message': 'Login successful', 'token': token, 'user_id': db_user['id']}), 200


@bp.before_app_request
def load_logged_in_user():
    """Load user from JWT token if provided."""
    g.user = None
    auth_header = request.headers.get('Authorization')
    
    if auth_header:
        try:
            # Extract token from "Bearer <token>"
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
                payload = jwt.decode(
                    token,
                    current_app.config.get('SECRET_KEY', 'dev'),
                    algorithms=['HS256']
                )
                user_id = payload.get('user_id')
                
                if user_id:
                    db = get_db()
                    g.user = db.execute(
                        'SELECT * FROM users WHERE id = ?', (user_id,)
                    ).fetchone()
        except jwt.ExpiredSignatureError:
            pass  # Token expired, no user loaded
        except jwt.InvalidTokenError:
            pass  # Invalid token, no user loaded


def login_required(view):
    """Decorator to require JWT authentication."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return jsonify({'error': 'Authentication required'}), 401
        return view(**kwargs)
    return wrapped_view


@bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """Logout the current user (JWT-based, client discards token)."""
    return jsonify({'message': 'Logged out successfully'}), 200
# logging endpoint for detections to backend
# api health endpoints