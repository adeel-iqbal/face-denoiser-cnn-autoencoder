import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load your trained autoencoder model
model = tf.keras.models.load_model("cnn_face_denoiser.keras")

IMG_SIZE = 128  # model expects 128x128 input

# Function to preprocess and denoise image
def denoise_image(input_img):
    # Convert image (PIL from Gradio) to numpy
    img = np.array(input_img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0  # normalize
    noisy = img + 0.2 * np.random.randn(*img.shape)  # add slight noise for demo
    noisy = np.clip(noisy, 0., 1.)

    # Predict denoised version
    denoised = model.predict(np.expand_dims(noisy, axis=0))[0]

    # Convert to displayable range
    return (noisy, denoised)

# Gradio interface
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        """
        # 🎭 Face Denoiser
        Upload a face image and watch the CNN Autoencoder remove noise in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Face Image")
            denoise_btn = gr.Button("Denoise Image", variant="primary")
        
        with gr.Column():
            noisy_output = gr.Image(label="Noisy Input")
            denoised_output = gr.Image(label="Denoised Output")
    
    denoise_btn.click(
        fn=denoise_image,
        inputs=input_image,
        outputs=[noisy_output, denoised_output]
    )
    
    gr.Markdown("*Powered by CNN Autoencoder trained on LFW dataset*")

if __name__ == "__main__":
    demo.launch()