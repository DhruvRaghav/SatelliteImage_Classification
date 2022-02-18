from flask import Flask, request, Response
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import Formatter
import json
import os
import keras
from keras.preprocessing import image
import numpy as np
from skimage import io
import cv2

if not (os.path.exists('Logs')):
    os.makedirs('Logs/',exist_ok=False)
log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
handler = TimedRotatingFileHandler('Logs/'+log_filename, when='MIDNIGHT', backupCount=7)

formatter = Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')

logger = logging.getLogger('gunicorn.error')

handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger.setLevel(logger.level)
logger.addHandler(handler)

logger.propagate = False

app = Flask(__name__)


try:
    model=keras.models.load_model('sat_classification_model.h5')
    classes = ['Dense_Residential', 'Sparse_Residential',
               'Medium_Residential', 'River', 'Forest', 'Highway', 'Industrial']

except Exception as e:
    logger.error(msg="model is not loaded")

@app.route('/sat_classification',methods=['POST'])  # Single Api
def classification():
    image1 = request.files['file']  # Single image path
    resp=Response(status=200,content_type='application/json')
    try:
        img=io.imread(image1)
        img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite('test.jpg',img2)
        result=''

        img = image.load_img('test.jpg', target_size=(400, 400, 3))
        img = image.img_to_array(img)
        img = img / 255
        proba = model.predict(img.reshape(1, 400, 400, 3))
        top_3 = np.argsort(proba[0])[:-6:-1]
        for i in range(5):
            if proba[0][top_3[i]] > 0.001:
                if(result==''):
                    result=str(classes[top_3[i]])
                else:
                    result=result+' , '+str(classes[top_3[i]])
        resp = json.dumps({'Detected classes': result})

        return resp
    except Exception as e:
        logger.error(msg=str(e))
        return "Error Occured", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=False)
