import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or 'your-secret-key-here'

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///math_solver.db'
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

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_funny_label(math_problem):
    """Generate a funny label for a math problem"""
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    
    try:
        if entry.image_url:
            os.remove(os.path.join(app.root_path, entry.image_url[1:]))
    except OSError:
        pass
    
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                img = Image.open(filepath)
                
                # First get the problem statement
                problem_prompt = """
                Extract just the math problem statement from this image.
                Return only the problem statement, no solution or working.
                """
                problem_response = model.generate_content([problem_prompt, img])
                problem_statement = problem_response.text
                
                # Generate funny label before solving
                funny_label = generate_funny_label(problem_statement)
                
                # Now solve the problem
                solution_prompt = """
                Solve this math problem step by step.
                Show all working clearly.
                Conclude with "Final Answer: [answer]"
                Remove all the stars and unnecessary comments from the reply
                
                """
                solution_response = model.generate_content([solution_prompt, img])
                solution = solution_response.text
                
                image_url = f"/static/uploads/{filename}"
                
                # Save to history with label
                history_entry = UserHistory(
                    user_id=current_user.id,
                    image_url=image_url,
                    solution=solution,
                    problem_label=funny_label
                )
                db.session.add(history_entry)
                db.session.commit()
                
                # Keep only last 10 entries per user
                entries = UserHistory.query.filter_by(user_id=current_user.id)\
                    .order_by(UserHistory.timestamp.desc())\
                    .offset(10)\
                    .all()
                for entry in entries:
                    try:
                        if entry.image_url:
                            os.remove(os.path.join(app.root_path, entry.image_url[1:]))
                    except OSError:
                        pass
                    db.session.delete(entry)
                db.session.commit()
                
                return jsonify({
                    'image_url': image_url,
                    'solution': solution,
                    'problem_label': funny_label
                })
            
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return render_template('index.html')

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
