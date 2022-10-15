import gradio as gr
from utils import CycleGAN, Predict
import glob


if __name__ == '__main__':

    Model = Predict(model=CycleGAN().gen_ptm, model_state='model/model.ckpt')

    app = gr.Interface(Model.predict,
                       gr.Image(type="pil", label='PHOTO').style(height=240),
                       gr.Image(type="pil", label='MONET STYLE').style(height=240),
                       allow_flagging='never',
                       examples=glob.glob("data/examples/*"))
    try:
        app.launch(server_name="0.0.0.0", server_port=7000)
    except KeyboardInterrupt:
        app.close()
    except Exception as e:
        print(e)
        app.close()

