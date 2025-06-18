# Skin Cancer Detection Using Machine Learning

This project is focused on classifying skin cancer types using Convolutional Neural Networks (CNN). A deep learning model is trained on dermatoscopic images to identify different categories of skin lesions. The application includes a web-based interface for users to upload an image and receive a prediction.

---

##  Technologies Used

- **Programming Language:** Python  
- **Libraries/Frameworks:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Flask  
- **Web Framework:** Flask (for deploying the model)  
- **IDE/Tools:** VS Code, Jupyter Notebook  
- **Version Control:** Git & GitHub

---

##  Dataset Details

- **Source:** ISIC - International Skin Imaging Collaboration  
- **Image Count:** ~25,000 labeled dermatoscopic images  
- **Classes:**
  - Melanoma  
  - Melanocytic nevus  
  - Basal cell carcinoma  
  - Actinic keratosis  
  - Benign keratosis  
  - Dermatofibroma  
  - Vascular lesions

---

##  Model Summary

- **Architecture:** Convolutional Neural Network (CNN)
- **Layers:** Conv2D → MaxPooling → Dropout → Flatten → Dense
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Accuracy:** ~93% on validation data

---



