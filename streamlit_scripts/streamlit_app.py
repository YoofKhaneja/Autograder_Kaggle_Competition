import streamlit as st
import numpy as np

import yaml
from essay_model import EssayModel
import pandas as pd



COLUMNS = ["cohesion","syntax","vocabulary","phraseology","grammar","convention"]
if "preds" not in st.session_state.keys():
    st.session_state["preds"] = None
def read_config(config_path):
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(config):
    cp_model = EssayModel.load_from_checkpoint(config["checkpoint_path"],config=config)
    return cp_model


def submit_essay():
    essay = st.session_state["essay"]
    if len(essay)!=0:
        prediction = model([essay])
        preds = np.round(prediction.cpu().detach().numpy(),decimals=2)
        preds_frame = pd.DataFrame(preds,columns=COLUMNS,index=None)
        st.session_state["preds"] = 1
        st.session_state["preds_frame"] = preds_frame
    else:
        st.error("Write an essay!")
    

def display_page(model=None):
    
    st.title("My ELL Essay Grader")
    
    st.text_area(label="my-essay",placeholder="Enter your essay here....",height=300,label_visibility="hidden",key="essay")
    st.button(label="Grade",on_click=submit_essay)
    
    if st.session_state["preds"]:
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(st.session_state["preds_frame"].style.format("{:.2f}"))


if __name__=="__main__":
    config = read_config("config.yaml")
    model = load_model(config)
    model.to(model.dev)
    model.eval()
    display_page(model)
    