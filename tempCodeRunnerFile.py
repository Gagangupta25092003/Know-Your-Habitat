        return "A5"
    elif a == 5:
        return "A6"
    elif a == 6:
        return "A7"
    elif a == 7:
        return "A8"
    elif a == 8:
        return "A9"
        
    
   
    
print(prediction_1(pred("A5.jpg")))

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('preict.html'))
if __name__ == '__main__':
    app.run()






