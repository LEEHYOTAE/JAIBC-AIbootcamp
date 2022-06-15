#!/usr/bin/env python
# coding: utf-8

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

import dill as pickle
with open('image_classifier_model.h5', 'rb') as f:
    clf = pickle.load(f)

st.write('Handwritten Number Recognizer')

CANVAS_SIZE = 192

col1, col2 = st.columns(2)
with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas'
    )

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    preview_img = cv2.resize(img,
        dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
    col2.image(preview_img)
    X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X = X.reshape((-1, 28, 28, 1))
    y = clf.predict(X)
    st.write('## Result: %d' % y)

