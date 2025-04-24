# Anime Face Creation using DCGAN (Keras + TensorFlow)

## 🧠 Problem Statement
Generating new, realistic anime-style human faces using Deep Convolutional Generative Adversarial Networks (DCGAN). The goal is to train a deep learning model capable of learning features of anime faces and synthesizing new ones that resemble real data.

## 📌 Explanation
This project implements a DCGAN (Deep Convolutional GAN) architecture using Keras and TensorFlow to generate anime faces. The generator network learns to create fake anime faces, while the discriminator tries to differentiate between real and generated ones. Through adversarial training, the generator improves to produce increasingly realistic results.

Key steps in the notebook:
- Data loading and preprocessing
- Building the Generator and Discriminator models
- Training using adversarial loss
- Visualizing generated samples during training

## 📂 Dataset
We used the "Anime Faces" dataset available on Kaggle:

[Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)

## 📽️ Output
The model progressively improves over training, producing high-quality anime face images. A selection of generated faces is shown at the end of training.
