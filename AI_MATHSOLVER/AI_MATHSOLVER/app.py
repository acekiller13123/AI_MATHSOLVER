import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

_instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(_instance_dir, exist_ok=True)

app = Flask(__name__, instance_relative_config=True)
app.secret_key = os.getenv('SECRET_KEY') or 'your-secret-key-here'

# Configure Database (SQLite under instance/)
_db_path = os.path.join(_instance_dir, 'math_solver.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + _db_path.replace('\\', '/')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# History Model with added problem_label column
class UserHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_url = db.Column(db.String(255))
    solution = db.Column(db.Text)
    problem_label = db.Column(db.String(100))  # New column for funny labels
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Solver: local ML (EasyOCR + SymPy) or Gemini API
GEMINI_API_KEY = (os.getenv('GEMINI_API_KEY') or '').strip()
SOLVER_BACKEND = (os.getenv('SOLVER_BACKEND') or 'auto').lower()
GEMINI_MODEL = os.getenv('GEMINI_MODEL') or 'gemini-2.0-flash'

if SOLVER_BACKEND not in ('local', 'auto', 'gemini'):
    raise ValueError("SOLVER_BACKEND must be one of: local, gemini, auto")
if SOLVER_BACKEND == 'gemini' and not GEMINI_API_KEY:
    raise ValueError(
        "SOLVER_BACKEND=gemini but GEMINI_API_KEY is not set. "
        "Set the key or use SOLVER_BACKEND=local."
    )

model = None
if SOLVER_BACKEND == 'gemini' or (SOLVER_BACKEND == 'auto' and GEMINI_API_KEY):
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Reject huge uploads early (avoids OCR / memory blowups)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

# DB column limit for history titles
_PROBLEM_LABEL_MAX = 100

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_funny_label(math_problem):
    """Generate a short humorous label (Gemini only)."""
    prompt = f"""
    This is a math problem: {math_problem[:200]}...
    Generate a very short (3-5 word), funny label for this problem that captures its essence in a humorous way.
    Examples:
    - "Pythagoras' Revenge"
    - "Algebraic Nightmare"
    - "Calculus Catastrophe"
    - "Fraction Frustration"
    - "Trigonometry Trauma"
    
    Just return the label, nothing else. Make it funny but appropriate.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip('"\' \n') or "Math Mystery"
    except Exception as e:
        print(f"Error generating label: {e}")
        return "Math Puzzle"


def _safe_remove_upload(relative_url: str) -> None:
    if not relative_url:
        return
    rel = relative_url.lstrip('/').replace('/', os.sep)
    full = os.path.normpath(os.path.join(app.root_path, rel))
    root = os.path.normpath(app.root_path)
    if not os.path.normcase(full).startswith(os.path.normcase(root)):
        return
    try:
        os.remove(full)
    except OSError:
        pass


def solve_uploaded_image(filepath: str, img: Image.Image) -> dict:
    """Returns problem_statement, solution, problem_label."""
    if model is not None:
        problem_prompt = """
                Extract just the math problem statement from this image.
                Return only the problem statement, no solution or working.
                """
        problem_response = model.generate_content([problem_prompt, img])
        problem_statement = problem_response.text or ""

        funny_label = generate_funny_label(problem_statement)

        solution_prompt = """
                Solve this math problem step by step.
                Show all working clearly.
                Conclude with "Final Answer: [answer]"
                Remove all the stars and unnecessary comments from the reply

                """
        solution_response = model.generate_content([solution_prompt, img])
        solution = solution_response.text or ""
        return {
            'problem_statement': problem_statement.strip(),
            'solution': solution,
            'problem_label': funny_label,
        }

    from ml.local_solver import solve_math_image

    out = solve_math_image(filepath)
    return {
        'problem_statement': out['problem_statement'],
        'solution': out['solution'],
        'problem_label': out['problem_label'],
    }


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    user_history = UserHistory.query.filter_by(user_id=current_user.id)\
        .order_by(UserHistory.timestamp.desc())\
        .limit(10)\
        .all()
    return jsonify([{
        'id': item.id,
        'image_url': item.image_url,
        'solution': item.solution,
        'problem_label': item.problem_label,
        'timestamp': item.timestamp.strftime('%Y-%m-%d %H:%M')
    } for item in user_history])

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    UserHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True})

@app.route('/delete_history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    entry = UserHistory.query.filter_by(id=history_id, user_id=current_user.id).first()
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404
    
    _safe_remove_upload(entry.image_url or '')
    
    db.session.delete(entry)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({'error': 'Invalid file name; use letters/numbers and a normal extension.'}), 400
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                img = Image.open(filepath)
                result = solve_uploaded_image(filepath, img)
                solution = result['solution']
                funny_label = (result.get('problem_label') or '')[:_PROBLEM_LABEL_MAX]
                problem_statement = result.get('problem_statement', '')

                image_url = f"/static/uploads/{filename}"

                history_entry = UserHistory(
                    user_id=current_user.id,
                    image_url=image_url,
                    solution=solution,
                    problem_label=funny_label,
                )
                db.session.add(history_entry)
                db.session.commit()

                entries = UserHistory.query.filter_by(user_id=current_user.id)\
                    .order_by(UserHistory.timestamp.desc())\
                    .offset(10)\
                    .all()
                for entry in entries:
                    _safe_remove_upload(entry.image_url or '')
                    db.session.delete(entry)
                db.session.commit()

                return jsonify({
                    'image_url': image_url,
                    'solution': solution,
                    'problem_label': funny_label,
                    'problem_statement': problem_statement,
                })
            
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        return jsonify({'error': 'Unsupported file type'}), 400
    
    return render_template('index.html')


@app.errorhandler(413)
def request_entity_too_large(_e):
    return jsonify({'error': 'Image too large (max 15 MB). Try a smaller file.'}), 413

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # threaded=True: long OCR/Gemini calls do not block other requests
    app.run(debug=True, threaded=True)
