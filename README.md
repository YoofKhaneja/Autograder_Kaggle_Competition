# ELL_Essay_Grader_Ayush_Yash_Aryaman
Final project for data mining by Ayush, Aryaman and Yash.<br>

We aim to build an automated essay grading system for ELL students from 8th grade to 12th grade. The dataset used is a part of a Kaggle Competition: The Feedback Prize - English Language Learning. Existing essay grading methods consist of models trained on essays written by students with english as their first language. As a result they are not able to aptly grade essays written by ELL students (students having English as their second language). This project aims to solve this problem by building a system that can successfully assess the writing of such ELL students.



## Project Structure
* Presentation: Consists of Presentation video and ppt
* notebooks: Consists of the classical models and transformer model notebooks
* streamlit_scripts: Consists of the scripts for the streamlit webapp

## Steps to run the streamlit scripts
* cd to the streamlit_scripts folder and install requirements using ``pip install -r requirements.txt```
* Download the models from https://drive.google.com/file/d/1j0iGjNqKyMsM0cftPT4_H3n0ljta9NGV/view?usp=share_link
* Unzip the folder to a folder named models in the strealit_scripts/ folder
* Open the streamlit_scripts folder in the terminal
* Run the code using ```streamlit run streamlit_app.py```
* The browser opens. Now type in the essay in the text box or copy it from Demo_Essays.pdf to test the model.
