import functools
import uuid
from flask import (
    Blueprint, flash, g, request, jsonify
)

from werkzeug.security import check_password_hash, generate_password_hash
from api.db import get_db

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=['POST', 'GET'])
def register():
    """ Admin should register new users"""
    user = request.get_json()
    if not user:
        return jsonify({'error': 'User information required'}), 400
    
    user_id = str(uuid.uuid4())
    db = get_db()
    db.execute(
        'INSERT INTO users (id, email, fullname, username, password, hashed_password) VALUES (?, ?, ?, ?, ?, ?)',
        (user_id, user['email'], user['fullname'], user['username'], generate_password_hash(user['password']))
    )
    db.commit()
    
    return '', 201

