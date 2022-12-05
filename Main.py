import numpy as np
import pickle
import cv2
from flask import  Flask, request, render_template
from werkzeug.utils import secure_filename
import os

with open("model-1","rb") as f:
    model = pickle.load(f)

def pred(s):
    test_image = cv2.imread(s)
    resize_image = cv2.resize(test_image,(256,256))
    resize_image = np.array(resize_image)
    resize_image = np.expand_dims(resize_image/255,0)
    output = model.predict(resize_image).tolist()
    output = output[0]
    ls = output.copy()
    ls.sort()
    print(output)
    print(ls)
    for i in range(0,9):
        if ls[-1] == output[i]:
            return i
        
def prediction_1(a):
    if a == 0:
        return "A10"
    elif a == 1:
        return "A13"
    elif a == 2:
        return "A15"
    elif a == 3:
        return "A17"
    elif a == 4:
        return "A5"
    elif a == 5:
        return "A6"
    elif a == 6:
        return "A7"
    elif a == 7:
        return "A8"
    elif a == 8:
        return "A9"
        
    
   
    
UPLOAD_FOLDER = os.path.join('static', 'uploads')


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/',methods =["GET", "POST"])
def main():
    if request.method == "POST":
        uploaded_img = request.files["img_name"]
        img_filename = secure_filename("image.jpg")
        gags = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
        uploaded_img.save(gags)
    
        print()
        tt = prediction_1(pred(gags))
        print(tt)
        print()
        return(render_template('Predicted.html', test= tt)) 
        
    else:
        return(render_template('predict.html'))   
if __name__ == '__main__':
   
    app.run()
    
    
    











