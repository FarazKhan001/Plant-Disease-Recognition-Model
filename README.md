# Plant Disease Recognition Model

An AI-powered web application that detects plant diseases from leaf images using Deep Learning, TensorFlow, Keras, and Flask.

## Project Overview

The Plant Disease Recognition System is designed to help farmers and agricultural workers identify plant diseases quickly and accurately using image classification.

The system uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset containing 87,000+ annotated leaf images across 38 disease categories and 14 crop types.

Users can upload a plant leaf image through a web interface, and the model predicts the disease along with its cause and treatment recommendations.

## Features

* Plant disease detection using Deep Learning
* Upload leaf images for prediction
* Real-time disease classification
* CNN model built using TensorFlow & Keras
* Flask-based web application
* Disease cause and treatment suggestions
* Responsive and user-friendly interface
* Animated frontend design with drag-and-drop support

## Technologies Used

### Frontend
* HTML
* CSS
* JavaScript

### Backend
* Python
* Flask

### AI / Machine Learning
* TensorFlow
* Keras
* NumPy

## Dataset

Dataset used: PlantVillage Dataset (Kaggle)

* 87,000+ leaf images
* 38 disease classes
* 14 crop types

### Crops Covered

* Apple
* Corn
* Grape
* Potato
* Tomato
* Strawberry
* Peach
* Soybean
* Pepper
* Cherry
* Orange
* Raspberry
* Squash
* Blueberry

## Disease Prediction Flow

1. User uploads a leaf image
2. Image is validated and resized
3. CNN model processes the image
4. Disease class is predicted
5. Disease information is fetched from JSON
6. Result is displayed with:

   * Disease Name
   * Cause
   * Treatment
     
## Results

* Successfully classifies 38 plant disease categories
* Real-time predictions within seconds
* Detects fungal, bacterial, and viral diseases
* Correctly identifies healthy plant conditions

## Future Scope

* Mobile Application using TensorFlow Lite
* Multi-disease detection
* Disease severity estimation
* Regional language support
* IoT and drone integration
* Transfer learning with ResNet50 / EfficientNet
  
## References

* PlantVillage Dataset (Kaggle)
* TensorFlow Documentation
* Keras Documentation
* Flask Documentation
* NumPy Documentation
