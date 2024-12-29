# **Next Word Prediction**

## **Project Overview**
The **Next Word Prediction** project is a Natural Language Processing (NLP) application that predicts the next word in a sequence of text. Using TensorFlow and Keras, this project builds and trains an LSTM-based sequential model to predict the most likely next word given an input sequence.

---

## **Workflow**
1. **Data Collection**:
   - A paragraph with multiple sentences is used as training data.

2. **Data Preprocessing**:
   - Libraries and packages like TensorFlow, Keras, and Tokenizer are used.
   - Tokenizer is applied to fit the text and create a word index.

3. **Sequence Preparation**:
   - Generated word sequences using `texts_to_sequences` for training.
   - Used `pad_sequences` to ensure consistent input dimensions.
   - Separated input and target words:
     - Input: Words except the last one in each sequence.
     - Target: The last word in each sequence.
   - Target labels are encoded using `to_categorical`.

4. **Model Building**:
   - Built a Sequential model using:
     - **Embedding Layer**: For mapping words to dense vectors.
     - **LSTM Layer**: To capture sequential dependencies.
     - **Dense Layer**: For prediction with a softmax activation.

5. **Model Compilation and Training**:
   - Compiled with loss (`categorical_crossentropy`), optimizer (`adam`), and metric (`accuracy`).
   - Trained on input sequences with a specified number of epochs.

6. **Model Testing**:
   - Used the trained model to predict the next word for a given sequence.

---

## **Technologies Used**
- **Programming Language**: Python
- **Environment**: Jupyter Notebook
- **Libraries and Frameworks**:
  - TensorFlow
  - Keras
  - NumPy
  - Tokenizer (from Keras Preprocessing)
  - Matplotlib (for visualization, optional)

---

## **Project Structure**
```
next-word-prediction/
│
├── README.md               # Project documentation
├── next_word_prediction.ipynb # Jupyter Notebook with complete implementation
├── dataset.txt             # Text data used for training (if applicable)
├── requirements.txt        # Required libraries and dependencies
└── models/
    ├── next_word_model.h5  # Saved trained model
    └── tokenizer.pkl       # Tokenizer object
```

---

## **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/next-word-prediction.git
   cd next-word-prediction
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

---

## **Usage**
1. **Training**:
   - Open the Jupyter Notebook `next_word_prediction.ipynb`.
   - Load the dataset and execute all the cells to train the model.
   - Save the model and tokenizer using `.h5` and `.pkl` formats.

2. **Testing**:
   - Provide an input sequence to the trained model.
   - The model will predict the most probable next word.

---

## **Model Architecture**
The model is built using TensorFlow and Keras:
```python
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 100, input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## **Features**
- **Preprocessing**:
  - Tokenization
  - Padding sequences
  - Categorical encoding
- **Model**:
  - Embedding for word representations.
  - LSTM for capturing word dependencies.
  - Softmax activation for multi-class classification.

---

## **Results**
- Achieved [Insert Accuracy]% accuracy on the training dataset after 100 epochs.
- Predictions are made with reasonable accuracy based on the input sequence.

---

## **Future Improvements**
- Experiment with different NLP models (e.g., Transformer-based architectures).
- Use a larger and more diverse dataset for better generalization.
- Integrate the model with a Flask/Django web application for deployment.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
