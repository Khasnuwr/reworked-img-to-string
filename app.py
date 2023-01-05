import streamlit as st
import cv2
import torch
import easyocr
from PIL import ImageFont, ImageDraw, Image, ImageGrab
import numpy as np


st.set_page_config(
    page_title="Numberplate Read and Detect System",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def read_str_from_img(frame):

    try:
        reader = easyocr.Reader(['bn'], gpu=True)
        result = reader.readtext(frame)
    except:
        reader = easyocr.Reader(['bn'], gpu=False)
        result = reader.readtext(frame)
    text = ''
    for string in result:
        text = text + string[1] + '\n'
    #print(text)
    return text






def plate_detection():
    path = 'best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
    frame_window = st.image([])
    capture = cv2.VideoCapture(0)
    scan = False
    destroy = False
    buttons = ['Read', 'Exit']
    for button in buttons:
        if st.button(button):
            if button == 'Read':
                #st.write(f'You pressed {button}')
                destroy = False
                scan = True

            if button == 'Exit':
                #st.write(f'You pressed {button}')
                destroy = True
                break

    while True:
        success, frame = capture.read()

        if not success:
            break

        else:
            results = model(frame)

            if scan:
                scan = False
                read_str_from_img(frame)
            if destroy:
                cv2.destroyAllWindows()
                capture.release()
                break

            frame = np.squeeze(results.render())


            frame = np.squeeze(results.render())




            # codes bellow commented for future development
            #
            # fontpath = "banglamn.ttc"  # <== 这里是宋体路径
            # font = ImageFont.truetype(fontpath, 24)
            # img_pil = Image.fromarray(frame)
            # draw = ImageDraw.Draw(img_pil)
            # b, g, r, a = 221, 82, 6, 0
            # draw.text((50, 80),  text, font=font, fill=(b, g, r, a))
            # frame = np.array(img_pil)

            ###############################################

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_window.image(frame)


            # STOP AND DESTROY CAPTURE FUNCTION


            # READ AND EXTRACTS THE TEXT FROM THE FRAME




    cv2.destroyAllWindows()
    capture.release()












# STARTS THE PLATE DETECTION FUNCTION AND HIDES THE "START" BUTTON


import streamlit as st

st.sidebar.title("MENU")
choice = st.sidebar.selectbox("Choose an option", ["Home", "Realtime detection", "Read from Picture"])

print(choice)

if choice == "Realtime detection":
    st.title("Numberplate Read and Detect System")

    plate_detection()

            
     



#st.title('SOMETHING WENT WRONG ;(')


elif choice == "Read from Picture":
    st.text("Upload Numberplate Picture to read 😎")
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)#, width=300)
        str_ = read_str_from_img(image)
        st.text(str_)
        
        try:
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

else:

    st.text("SELECT READ NUMBERPLATE TO READ 😊")
