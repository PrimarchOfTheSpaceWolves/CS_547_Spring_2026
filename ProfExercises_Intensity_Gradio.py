import numpy as np
import cv2
import matplotlib.pyplot as plt
import gradio as gr

def create_transform_plot(transform, title="Intensity Transform"):
    fig, subfig = plt.subplots(1, 1, figsize=(5,5))
    x = np.arange(256)
    line = subfig.plot(x, transform, color="black", linewidth=1)
    fill = subfig.fill_between(x, transform, color="gray", alpha=0.5)
    subfig.set_xlabel("Input Intensity")
    subfig.set_ylabel("Ouput Intensity")
    subfig.set_xlim([0,255])
    subfig.set_ylim([0,255])
    subfig.set_title(title)
    plt.close(fig)
    return fig

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                color_image = gr.Image(label="Input Image")
            with gr.Column():
                gray_image = gr.Image(label="Grayscale Image")
                one_button = gr.Button("MAGIC")
        with gr.Row():
            with gr.Column():
                hist_plot = gr.Plot(label="My Histogram")
                
        def do_magic(gray):
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            gray = np.where(gray > 200, 255, 0)
            return gray
                
        one_button.click(fn=do_magic, inputs=gray_image, outputs=gray_image)
        
        def on_upload(color):
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            plot = create_transform_plot(np.arange(256, dtype="uint8"))            
            return gray, plot
        
        color_image.upload(fn=on_upload, 
                           inputs=[color_image], 
                           outputs=[gray_image, hist_plot])
                
    demo.launch()

if __name__ == "__main__":
    main()
    