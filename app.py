import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the trained model
model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Prediction function
def prediction(image_path):
    # Open the image
    image = Image.open(image_path)
    image=image.convert('RGB')  
    

    # Resize the image to 224x224
    image = image.resize((224, 224))

    # Convert the image to a tensor
    input_data = TF.to_tensor(image)

    # Add batch dimension [1, 3, 224, 224] (model expects 4D tensor: [batch, channels, height, width])
    input_data = input_data.unsqueeze(0)

    # Pass the image through the model to get the prediction
    output = model(input_data)

    # Convert the output to a numpy array and get the predicted class index
    output = output.detach().numpy()
    index = np.argmax(output)

    return index


# Flask app setup
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/ai-engine')
def ai_engine_page():
    return render_template('aiengine.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        # Predict disease
        pred = prediction(file_path)
        
        # Retrieve disease details from the dataset
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        # Render the result page with prediction details
        return render_template('final.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
