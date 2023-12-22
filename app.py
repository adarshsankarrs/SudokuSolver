from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import os, sys, cv2
import numpy as np
sys.path.insert(1, './bin/')
import sudoku_extractor as sud

WORKING_DIRECTORY = './static/working-dir/'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and len(request.form)==0:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            # On the website, the extracted sudoku, along with 
            # and editable matrix with the values is shown.
            (board, warped_img) = sud.sudoku_extractor(img_path)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "warped_image.png"), warped_img)
            return render_template('index.html', state=1, arr=board)
    elif request.method == 'POST':
        corrected_board=np.zeros((9,9))
        # user can edit/ verify the extracted sudoku, and return
        # it is then sent to solve.
        for elem in request.form:
            index=elem[-2:]
            corrected_board[int(index[0]),int(index[1])]=int(request.form.get(elem))
        solved_board=sud.solver(corrected_board.astype(int))
        sud.show_solution(solved_board)
        return render_template('index.html', state=3, arr=None)
    else:
        if os.listdir(WORKING_DIRECTORY) != []:
            for f in os.listdir(WORKING_DIRECTORY):
                if f != "temp.txt":
                    os.remove(os.path.join(WORKING_DIRECTORY, f))
        return render_template('index.html', state=0, arr=None)


app.secret_key = 'cb1501f35e034aa18dd6c3743f4363bb'
app.config['UPLOAD_FOLDER'] = WORKING_DIRECTORY
app.run(host ='0.0.0.0', port = 5001) 