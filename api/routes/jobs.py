"""
Job queue endpoints.
Provides endpoints for queuing and monitoring video processing jobs.
"""
import json
import csv
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from io import StringIO, BytesIO
import cv2
from flask import Blueprint, jsonify, request, current_app, send_file, after_this_request
from ..jobs import create_job, get_job_status, get_jobs_by_date_and_time
from ..db import get_db

bp = Blueprint('jobs', __name__, url_prefix='/jobs')


def _ffmpeg_executable():
    """
    Return ffmpeg binary path if available (browser-safe clips need H.264).

    Resolution order:
      1. FFMPEG_PATH / FFMPEG_BINARY environment variables (full path to ffmpeg.exe)
      2. Flask app.config['FFMPEG_PATH']
      3. shutil.which('ffmpeg' / 'ffmpeg.exe') on the current process PATH
      4. Common Windows install locations (IDE-started Python often lacks user PATH)
    """
    candidates = []

    def _add(p):
        if not p:
            return
        s = str(p).strip().strip('"')
        if not s:
            return
        candidates.append(os.path.normpath(s))

    _add(os.environ.get('FFMPEG_PATH'))
    _add(os.environ.get('FFMPEG_BINARY'))
    try:
        _add(current_app.config.get('FFMPEG_PATH'))
    except RuntimeError:
        pass

    for name in ('ffmpeg', 'ffmpeg.exe'):
        w = shutil.which(name)
        if w:
            _add(w)

    if os.name == 'nt':
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        program_files_x86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
        local_app = os.environ.get('LOCALAPPDATA', '')
        _add(os.path.join(program_files, 'ffmpeg', 'bin', 'ffmpeg.exe'))
        _add(os.path.join(program_files, 'Gyan', 'FFmpeg', 'bin', 'ffmpeg.exe'))
        _add(os.path.join(program_files_x86, 'ffmpeg', 'bin', 'ffmpeg.exe'))
        _add(r'C:\ffmpeg\bin\ffmpeg.exe')
        _add(os.path.join(local_app, 'Microsoft', 'WinGet', 'Links', 'ffmpeg.exe'))
        choco_bin = os.path.join(
            os.environ.get('ChocolateyInstall', r'C:\ProgramData\chocolatey'),
            'bin',
            'ffmpeg.exe',
        )
        _add(choco_bin)

        pkg_root = os.path.join(local_app, 'Microsoft', 'WinGet', 'Packages')
        if os.path.isdir(pkg_root):
            try:
                for entry in os.scandir(pkg_root):
                    if not entry.is_dir():
                        continue
                    if 'ffmpeg' not in entry.name.lower():
                        continue
                    for rel in (
                        ('bin', 'ffmpeg.exe'),
                        ('ffmpeg', 'bin', 'ffmpeg.exe'),
                    ):
                        _add(os.path.join(entry.path, *rel))
            except OSError:
                pass

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def _ffmpeg_extract_h264_clip(video_path: str, start_sec: float, duration_sec: float, out_path: str) -> bool:
    """
    Cut a segment and encode as H.264 + yuv420p in MP4 with faststart.
    OpenCV's mp4v/MPEG-4 Part 2 output often will not play in HTML5 <video>.
    """
    exe = _ffmpeg_executable()
    if not exe:
        return False
    if duration_sec <= 0:
        return False
    cmd = [
        exe,
        '-y',
        '-hide_banner',
        '-loglevel',
        'error',
        '-ss',
        str(max(0.0, start_sec)),
        '-i',
        video_path,
        '-t',
        str(duration_sec),
        '-vf',
        'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '23',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-an',
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            current_app.logger.warning(
                'ffmpeg clip failed: %s', (result.stderr or result.stdout or 'unknown')[:500]
            )
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        current_app.logger.warning('ffmpeg clip exception: %s', e)
        return False


def _ffmpeg_transcode_to_browser_mp4(src_path: str, out_path: str) -> bool:
    """Re-encode any video readable by ffmpeg to browser-safe H.264 MP4."""
    exe = _ffmpeg_executable()
    if not exe:
        return False
    cmd = [
        exe,
        '-y',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        src_path,
        '-vf',
        'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '23',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-an',
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            current_app.logger.warning(
                'ffmpeg transcode failed: %s',
                (result.stderr or result.stdout or 'unknown')[:500],
            )
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        current_app.logger.warning('ffmpeg transcode exception: %s', e)
        return False


def _opencv_write_raw_clip(
    video_path: str, clip_start: float, clip_end: float, out_path: str
) -> int:
    """Write frames between clip_start and clip_end using OpenCV (any codec). Returns frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return 0
    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000.0)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_sec > clip_end:
                break
            writer.write(frame)
            frames_written += 1
    finally:
        cap.release()
        writer.release()
    return frames_written


@bp.route('', methods=['POST'])
def queue_video_job():
    """
    Queue a video for background processing.
    
    Request body (JSON):
    {
        'video_data': '<base64-encoded video>',  // OR upload as multipart file 'video'
        'query_image': '<base64-encoded query image>',  // OR upload as multipart file 'query_image'
        'camera_id': '<camera ID>',
        'detection_id': '<optional detection_id to associate results>',
        'threshold': <optional similarity threshold, default 40>,
        'frame_skip': <optional frames to skip, default 15>
    }
    """
    try:
        # Validate camera_id
        camera_id = request.json.get('camera_id') if request.json else None
        if not camera_id:
            return jsonify({'error': 'camera_id is required'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (camera_id,)).fetchone()
        if camera is None:
            return jsonify({'error': f'Camera {camera_id} not found'}), 404
        
        # Get processing parameters
        # we would also want to select a time range and and a camera location,
        # parameter to get from frontend, multiple images can have single job
        threshold = request.json.get('threshold', 40) if request.json else 40
        frame_skip = request.json.get('frame_skip', 15) if request.json else 15
        detection_id = request.json.get('detection_id') if request.json else None
        video_filename = request.json.get('video_filename') if request.json else None
        query_image_filename = request.json.get('query_image_filename') if request.json else None
        
        if not video_filename or not query_image_filename:
            return jsonify({'error': 'video_filename and query_image_filename are required'}), 400
        
        # Queue job
        job_id = create_job(
            camera_id=camera_id,
            detection_id=detection_id,
            video_filename=video_filename,
            query_image_filename=query_image_filename,
            threshold=threshold,
            frame_skip=frame_skip
        )
        
        # TODO: return name of file being processed
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video queued for processing',
            'detection_id': detection_id,
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error queuing video: {str(e)}")
        return jsonify({'error': f'Failed to queue video: {str(e)}'}), 500


@bp.route('/<job_id>', methods=['GET'])
def get_job_status_route(job_id):
    """
    Get the status of a queued video processing job.
    
    Returns:
    {
        'id': '<job_id>',
        'status': 'pending|processing|completed|failed',
        'created_at': '<timestamp>',
        'started_at': '<timestamp or null>',
        'completed_at': '<timestamp or null>',
        'results': { ... } // only if completed
        'error': '<error message>' // only if failed
    }
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': f'Failed to get job status: {str(e)}'}), 500


@bp.route('/<job_id>/cancel', methods=['PATCH'])
def cancel_job(job_id):
    """
    Cancel a pending or processing job.

    Only jobs in 'pending' or 'processing' state can be cancelled.
    The worker polls the DB between iterations and will stop as soon
    as it sees the 'cancelled' status.
    """
    try:
        db = get_db()
        job = db.execute('SELECT id, status FROM jobs WHERE id = ?', (job_id,)).fetchone()

        if job is None:
            return jsonify({'error': 'Job not found'}), 404

        if job['status'] not in ('pending', 'processing'):
            return jsonify({
                'error': f"Cannot cancel a job that is already '{job['status']}'"
            }), 400

        db.execute(
            "UPDATE jobs SET status = 'cancelled', completed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), job_id)
        )
        db.commit()

        current_app.logger.info(f"Job {job_id} cancelled by user")
        return jsonify({'job_id': job_id, 'status': 'cancelled'}), 200

    except Exception as e:
        current_app.logger.error(f"Error cancelling job: {str(e)}")
        return jsonify({'error': f'Failed to cancel job: {str(e)}'}), 500


@bp.route('/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """
    Delete a job and its associated files.

    Can delete jobs in any state. Removes job record from database
    and cleans up associated upload files if they exist.
    """
    try:
        from ..jobs.models import get_job_query_images
        
        db = get_db()
        job = db.execute(
            'SELECT id, status, video_filename, query_image_filename FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()

        if job is None:
            return jsonify({'error': 'Job not found'}), 404

        # Delete associated files from uploads directory
        uploads_dir = os.path.join(current_app.instance_path, 'uploads')
        files_to_delete = []
        
        # Add single query image if it exists (for regular jobs)
        if job['video_filename']:
            files_to_delete.append(job['video_filename'])
        if job['query_image_filename']:
            files_to_delete.append(job['query_image_filename'])
        
        # For batch jobs, also get query images from job_query_images table
        try:
            batch_query_images = get_job_query_images(job_id)
            files_to_delete.extend(batch_query_images)
        except Exception as e:
            current_app.logger.warning(f"Could not fetch batch query images for job {job_id}: {str(e)}")
        
        # Delete all collected files
        for filename in set(files_to_delete):  # Use set to avoid duplicates
            if not filename:
                continue
            file_path = os.path.join(uploads_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    current_app.logger.info(f"Deleted file: {file_path}")
                except OSError as e:
                    current_app.logger.warning(f"Could not delete file {file_path}: {str(e)}")

        # Delete the job from database (this cascades to job_query_images due to FK constraint)
        db.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
        db.commit()

        current_app.logger.info(f"Job {job_id} deleted by user")
        return jsonify({'job_id': job_id, 'message': 'Job deleted successfully'}), 200

    except Exception as e:
        current_app.logger.error(f"Error deleting job: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to delete job: {str(e)}'}), 500


@bp.route('/<job_id>/rerun', methods=['POST'])
def rerun_job(job_id):
    """
    Rerun a completed or failed job with the same parameters.

    Creates a new job with the same camera, threshold, frame_skip,
    and other parameters as the original job. Can rerun jobs in
    'completed', 'failed', or 'cancelled' states.
    
    Handles both regular jobs and batch jobs (with multiple query images).
    """
    try:
        from ..jobs.models import get_job_query_images, create_batch_job, create_job
        
        db = get_db()
        original_job = db.execute(
            '''SELECT id, camera_id, threshold, frame_skip, video_filename,
                      query_image_filename, detection_id, job_date, start_time, end_time
               FROM jobs WHERE id = ?''',
            (job_id,)
        ).fetchone()

        if original_job is None:
            return jsonify({'error': 'Job not found'}), 404

        # Check if this is a batch job (has no video_filename)
        is_batch_job = original_job['video_filename'] is None
        
        if is_batch_job:
            # For batch jobs, fetch all query images
            query_images = get_job_query_images(job_id)
            if not query_images:
                return jsonify({'error': 'Batch job has no query images to rerun'}), 400
            
            # Extract filenames from Row objects - handle both dict-like and tuple access
            query_image_filenames = []
            for img in query_images:
                try:
                    # Try dictionary-style access first
                    filename = img['query_image_filename']
                except (TypeError, KeyError):
                    try:
                        # Fall back to tuple-style access (index 0)
                        filename = img[0]
                    except (TypeError, IndexError):
                        current_app.logger.warning(f"Could not extract filename from: {img}")
                        continue
                query_image_filenames.append(filename)
            
            if not query_image_filenames:
                return jsonify({'error': 'Could not extract query image filenames'}), 400
            
            # Create a new batch job with the same parameters
            new_job_id = create_batch_job(
                camera_id=original_job['camera_id'],
                query_image_filenames=query_image_filenames,
                threshold=original_job['threshold'],
                frame_skip=original_job['frame_skip'],
                job_date=original_job['job_date'],
                start_time=original_job['start_time'],
                end_time=original_job['end_time']
            )
        else:
            # For regular jobs, use create_job
            new_job_id = create_job(
                camera_id=original_job['camera_id'],
                detection_id=original_job['detection_id'],
                video_filename=original_job['video_filename'],
                query_image_filename=original_job['query_image_filename'],
                threshold=original_job['threshold'],
                frame_skip=original_job['frame_skip'],
                job_date=original_job['job_date'],
                start_time=original_job['start_time'],
                end_time=original_job['end_time']
            )

        current_app.logger.info(f"Job {job_id} rerun as new job {new_job_id}")
        return jsonify({
            'new_job_id': new_job_id,
            'status': 'pending',
            'message': f'Job rerun successfully. New job ID: {new_job_id}'
        }), 202

    except Exception as e:
        current_app.logger.error(f"Error rerunning job: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to rerun job: {str(e)}'}), 500


@bp.route('', methods=['GET'])
def list_jobs():
    """
    Get all jobs with optional filtering.
    
    Query parameters:
        status: Filter by status (pending, processing, completed, failed)
        limit: Number of jobs to return (default 100, max 500)
        offset: Pagination offset (default 0)
    """
    try:
        status_filter = request.args.get('status')
        limit = min(request.args.get('limit', 100, type=int), 500)
        offset = request.args.get('offset', 0, type=int)
        
        db = get_db()
        
        query = (
            'SELECT id, status, camera_id, detection_id, '
            'query_image_filename, threshold, frame_skip, '
            'job_date, start_time, end_time, '
            'result_data, error_message, '
            'created_at, started_at, completed_at '
            'FROM jobs'
        )
        params = []
        
        if status_filter:
            query += ' WHERE status = ?'
            params.append(status_filter)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        jobs = db.execute(query, params).fetchall()
        
        job_list = []
        for job in jobs:
            job_dict = dict(job)
            # Provide a human-readable file_name for the frontend.
            # Batch jobs have their filenames in job_query_images; expose a
            # placeholder here so the frontend always gets a non-null value.
            if job_dict.get('query_image_filename'):
                job_dict['file_name'] = job_dict['query_image_filename']
            else:
                # Batch job – fetch first query image filename as label
                first = db.execute(
                    'SELECT query_image_filename FROM job_query_images '
                    'WHERE job_id = ? ORDER BY created_at LIMIT 1',
                    (job_dict['id'],)
                ).fetchone()
                job_dict['file_name'] = first['query_image_filename'] if first else 'Batch Job'
            # Expose lightweight in-progress fields; strip heavy result_data (e.g. completed matches).
            raw_result = job_dict.get('result_data')
            if raw_result and job_dict.get('status') == 'processing':
                try:
                    pdata = json.loads(raw_result)
                    if isinstance(pdata, dict):
                        for key in (
                            'progress_percent',
                            'frames_total',
                            'frames_read',
                            'frames_processed',
                        ):
                            if key in pdata and pdata[key] is not None:
                                job_dict[key] = pdata[key]
                except (json.JSONDecodeError, TypeError):
                    pass
            job_dict.pop('result_data', None)
            job_list.append(job_dict)
        
        return jsonify({
            'jobs': job_list,
            'count': len(job_list),
            'limit': limit,
            'offset': offset,
            'filter_status': status_filter
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing jobs: {str(e)}")
        return jsonify({'error': f'Failed to list jobs: {str(e)}'}), 500


@bp.route('/search/by-date-time', methods=['GET'])
def search_jobs_by_date_time():
    """
    Search for jobs by date and time range.
    
    Query parameters:
        camera_id: Optional camera ID filter
        job_date: Optional date in YYYY-MM-DD format
        start_time: Optional start time in HH:MM:SS format
        end_time: Optional end time in HH:MM:SS format
        limit: Number of jobs to return (default 100, max 500)
        offset: Pagination offset (default 0)
    
    Returns:
        List of jobs matching the criteria
    """
    try:
        camera_id = request.args.get('camera_id')
        job_date = request.args.get('job_date')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = min(request.args.get('limit', 100, type=int), 500)
        offset = request.args.get('offset', 0, type=int)
        
        # At least one filter should be specified
        if not any([camera_id, job_date, start_time, end_time]):
            return jsonify({
                'error': 'At least one filter (camera_id, job_date, start_time, or end_time) is required'
            }), 400
        
        jobs = get_jobs_by_date_and_time(
            camera_id=camera_id,
            job_date=job_date,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'jobs': jobs,
            'count': len(jobs),
            'limit': limit,
            'offset': offset,
            'filters': {
                'camera_id': camera_id,
                'job_date': job_date,
                'start_time': start_time,
                'end_time': end_time
            }
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error searching jobs by date/time: {str(e)}")
        return jsonify({'error': f'Failed to search jobs: {str(e)}'}), 500


@bp.route('/<job_id>/results/json', methods=['GET'])
def get_job_results_json(job_id):
    """
    Download raw results as JSON.
    
    Returns:
        JSON file with raw results from the completed job.
        Only available for completed jobs.
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': f"Results not available for job in '{job['status']}' status"
            }), 400
        
        db = get_db()
        result = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        
        if not result or not result['result_data']:
            return jsonify({'error': 'No results found for this job'}), 404
        
        result_data = json.loads(result['result_data'])
        
        # Create BytesIO object for file download
        json_bytes = BytesIO(
            json.dumps(result_data, indent=2).encode('utf-8')
        )
        json_bytes.seek(0)
        
        return send_file(
            json_bytes,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'job_{job_id}_results.json'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error downloading job results as JSON: {str(e)}")
        return jsonify({'error': f'Failed to download results: {str(e)}'}), 500


@bp.route('/<job_id>/results/csv', methods=['GET'])
def get_job_results_csv(job_id):
    """
    Download raw results as CSV.
    
    Returns:
        CSV file with results from the completed job.
        Only available for completed jobs.
        Flattens nested JSON structures.
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': f"Results not available for job in '{job['status']}' status"
            }), 400
        
        db = get_db()
        result = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        
        if not result or not result['result_data']:
            return jsonify({'error': 'No results found for this job'}), 404
        
        result_data = json.loads(result['result_data'])
        
        # Helper function to flatten nested structures
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary."""
            items = []
            if isinstance(d, dict):
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list):
                        # Convert lists to JSON string
                        items.append((new_key, json.dumps(v)))
                    else:
                        items.append((new_key, v))
            elif isinstance(d, list):
                # If root is a list, process each item
                for i, item in enumerate(d):
                    flattened = flatten_dict(item, '', sep=sep)
                    if i == 0:
                        items.extend(flattened.items())
                    else:
                        break
            return dict(items)
        
        # Handle both dict and list results
        if isinstance(result_data, list):
            # List of detections/matches
            rows = [flatten_dict(item) for item in result_data]
        else:
            # Single result object — check if there's a 'matches' key
            matches = result_data.get('matches', None)
            if matches is not None:
                rows = [flatten_dict(item) for item in matches]
            else:
                rows = [flatten_dict(result_data)]

        # Strip frame_image from every row — it is a large base64 blob that
        # has no value in a CSV and would make the file unreadable.
        for row in rows:
            row.pop('frame_image', None)
        
        if not rows:
            return jsonify({'error': 'No data to export'}), 400
        
        # Create CSV in memory
        csv_buffer = StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
        
        csv_bytes = BytesIO(csv_buffer.getvalue().encode('utf-8'))
        csv_bytes.seek(0)
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'job_{job_id}_results.csv'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error downloading job results as CSV: {str(e)}")
        return jsonify({'error': f'Failed to download results: {str(e)}'}), 500


@bp.route('/<job_id>/results/matches', methods=['GET'])
def get_job_matches(job_id):
    """
    Return the matches array for a completed job as a JSON response.

    This includes the 'frame_image' base64 field on each match so the
    frontend can render vehicle frame thumbnails without re-processing
    the video.

    Returns:
    {
        'job_id': '<job_id>',
        'total_matches': int,
        'matches': [
            {
                'time': float,           // timestamp in seconds
                'match_percent': float,  // similarity as 0-100
                'frame_id': int,
                'box': { x1, y1, x2, y2 },
                'frame_image': 'data:image/jpeg;base64,...'  // JPEG crop
            },
            ...
        ]
    }
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404

        if job['status'] != 'completed':
            return jsonify({
                'error': f"Results not available for job in '{job['status']}' status"
            }), 400

        db = get_db()
        result = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()

        if not result or not result['result_data']:
            return jsonify({'error': 'No results found for this job'}), 404

        result_data = json.loads(result['result_data'])
        matches = result_data.get('matches', [])

        return jsonify({
            'job_id': job_id,
            'total_matches': len(matches),
            'matches': matches,  # frame_image included per match
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching job matches: {str(e)}")
        return jsonify({'error': f'Failed to fetch matches: {str(e)}'}), 500


@bp.route('/<job_id>/results/clip', methods=['GET'])
def get_job_match_clip(job_id):
    """
    Build and return a short MP4 clip around a match timestamp.

    Query params:
        time: float seconds in source video (required)
        duration: float clip length in seconds (optional, default=5.0, max=5.0)
        video_filename: source video filename (optional; required for batch jobs)
    """
    temp_clip_path = None
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        if job['status'] != 'completed':
            return jsonify({
                'error': f"Clip not available for job in '{job['status']}' status"
            }), 400

        timestamp = request.args.get('time', type=float)
        if timestamp is None:
            return jsonify({'error': 'Query parameter "time" is required'}), 400
        if timestamp < 0:
            return jsonify({'error': 'Query parameter "time" must be >= 0'}), 400

        duration = request.args.get('duration', default=5.0, type=float)
        if duration is None:
            duration = 5.0
        duration = max(0.2, min(5.0, duration))

        db = get_db()
        requested_video_filename = request.args.get('video_filename', type=str)
        job_row = db.execute(
            'SELECT video_filename FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()

        video_filename = requested_video_filename or (job_row['video_filename'] if job_row else None)
        if not video_filename:
            return jsonify({'error': 'video_filename is required for this job clip'}), 400

        uploads_dir = os.path.join(current_app.instance_path, 'uploads')
        video_path = os.path.join(uploads_dir, video_filename)
        if not os.path.exists(video_path):
            video_row = db.execute(
                'SELECT storage_path FROM videos WHERE filename = ? ORDER BY captured_at DESC LIMIT 1',
                (video_filename,)
            ).fetchone()
            if video_row and video_row['storage_path'] and os.path.exists(video_row['storage_path']):
                video_path = video_row['storage_path']
            else:
                return jsonify({'error': f'Video file not found: {video_filename}'}), 404

        cap_probe = cv2.VideoCapture(video_path)
        if not cap_probe.isOpened():
            return jsonify({'error': 'Could not open source video'}), 500

        fps = cap_probe.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0

        width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap_probe.release()
            return jsonify({'error': 'Invalid source video dimensions'}), 500

        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = (total_frames / fps) if total_frames > 0 else None
        cap_probe.release()

        clip_start = max(0.0, timestamp - (duration / 2.0))
        clip_end = clip_start + duration
        if total_seconds is not None and total_seconds > 0:
            clip_end = min(clip_end, total_seconds)
            if clip_end <= clip_start:
                clip_start = max(0.0, clip_end - duration)

        clip_span = clip_end - clip_start
        if clip_span <= 0:
            return jsonify({'error': 'Could not build clip at this timestamp'}), 404

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_clip:
            temp_clip_path = tmp_clip.name

        ok_ffmpeg = _ffmpeg_extract_h264_clip(video_path, clip_start, clip_span, temp_clip_path)

        if not ok_ffmpeg:
            raw_suffix = '_raw.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=raw_suffix) as tmp_raw:
                temp_raw_path = tmp_raw.name
            try:
                frames_written = _opencv_write_raw_clip(
                    video_path, clip_start, clip_end, temp_raw_path
                )
                if frames_written == 0:
                    if os.path.exists(temp_clip_path):
                        os.remove(temp_clip_path)
                    return jsonify({'error': 'Could not build clip at this timestamp'}), 404
                if not _ffmpeg_executable():
                    if os.path.exists(temp_clip_path):
                        os.remove(temp_clip_path)
                    return jsonify({
                        'error': (
                            'Clip preview needs ffmpeg with libx264. The server process could not '
                            'find ffmpeg.exe. Install FFmpeg for Windows, then either add its bin '
                            'folder to the system PATH and restart the backend, or set the '
                            'environment variable FFMPEG_PATH to the full path of ffmpeg.exe '
                            '(for example C:\\ffmpeg\\bin\\ffmpeg.exe).'
                        )
                    }), 503
                if not _ffmpeg_transcode_to_browser_mp4(temp_raw_path, temp_clip_path):
                    if os.path.exists(temp_clip_path):
                        os.remove(temp_clip_path)
                    return jsonify({
                        'error': 'Could not encode clip for browser playback (ffmpeg libx264 failed).'
                    }), 500
            finally:
                if os.path.exists(temp_raw_path):
                    try:
                        os.remove(temp_raw_path)
                    except OSError:
                        pass

        if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
            if temp_clip_path and os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
            return jsonify({'error': 'Could not build clip at this timestamp'}), 404

        @after_this_request
        def _cleanup_temp_clip(response):
            try:
                if temp_clip_path and os.path.exists(temp_clip_path):
                    os.remove(temp_clip_path)
            except Exception:
                pass
            return response

        return send_file(
            temp_clip_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f'job_{job_id}_clip_{int(timestamp)}.mp4'
        )

    except Exception as e:
        if temp_clip_path and os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except Exception:
                pass
        current_app.logger.error(f"Error generating job match clip: {str(e)}")
        return jsonify({'error': f'Failed to generate clip: {str(e)}'}), 500
