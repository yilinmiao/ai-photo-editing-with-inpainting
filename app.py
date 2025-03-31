import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 512


def generate_app(get_processed_inputs, inpaint):
    def get_points(img, evt: gr.SelectData, state):
        # If state is None, initialize it
        if state is None:
            state = {"input_image": img.copy(), "input_points": []}

        # If this is the first point, save the original image
        if len(state["input_points"]) == 0:
            state["input_image"] = img.copy()

        # Add the selected point
        state["input_points"].append([evt.index[0], evt.index[1]])

        # Mark selected points with a green crossmark on the displayed image
        display_img = img.copy()
        draw = ImageDraw.Draw(display_img)
        size = 10
        for point in state["input_points"]:
            x, y = point
            draw.line((x - size, y, x + size, y), fill="green", width=5)
            draw.line((x, y - size, x, y + size), fill="green", width=5)

        # Run SAM to get mask
        try:
            mask = get_processed_inputs(state["input_image"], [state["input_points"]])
            res_mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))

            sam_output = (
                state["input_image"].resize((IMG_SIZE, IMG_SIZE)),
                [
                    (res_mask, "background"),
                    (~res_mask, "subject")
                ]
            )
        except Exception as e:
            raise gr.Error(f"SAM error: {str(e)}")

        return display_img, sam_output, state

    def run_inpaint(state, prompt, negative_prompt, cfg, seed, invert):
        if state is None or "input_points" not in state or len(state["input_points"]) == 0:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

        try:
            # Get mask from state
            mask = get_processed_inputs(state["input_image"], [state["input_points"]])
            res_mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))

            # Apply inversion if requested
            if bool(invert):
                what = 'subject'
                res_mask = ~res_mask
            else:
                what = 'background'

            gr.Info(f"Inpainting {what}... (this will take up to a few minutes)")

            # Run inpainting
            inpainted = inpaint(state["input_image"], res_mask, prompt, negative_prompt, seed, cfg)
            return inpainted.resize((IMG_SIZE, IMG_SIZE))

        except Exception as e:
            raise gr.Error(f"Inpainting error: {str(e)}")

    def reset_state():
        return None

    def preprocess_image(img):
        if img is None:
            return None

        # Make sure the image is square
        width, height = img.size

        if width != height:
            gr.Warning("Image is not square, adding white padding")
            # Determine the size for the new square image
            new_size = max(width, height)
            # Create a new image with white background
            new_image = Image.new("RGB", (new_size, new_size), 'white')
            # Calculate position to paste the original image
            left = (new_size - width) // 2
            top = (new_size - height) // 2
            # Paste the original image onto the new image
            new_image.paste(img, (left, top))
            img = new_image

        return img.resize((IMG_SIZE, IMG_SIZE))

    def load_example(img_path, prompt_text, neg_prompt_text, seed_value):
        if img_path is None:
            return None, prompt_text, neg_prompt_text, seed_value, None

        img = Image.open(img_path).convert("RGB")
        processed_img = preprocess_image(img)
        return processed_img, prompt_text, neg_prompt_text, seed_value, None

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Image inpainting
            1. Upload an image by clicking on the first canvas.
            2. Click on the subject you would like to keep. Immediately SAM will be run and you will see the results. If you
               are happy with those results move to the next step, otherwise add more points to refine your mask.
            3. Write a prompt (and optionally a negative prompt) for what you want to generate for the infilling. 
               Adjust the CFG scale and the seed if needed. You can also invert the mask, i.e., infill the subject 
               instead of the background by toggling the relative checkmark.
            4. Click on "run inpaint" and wait for up to two minutes. If you are not happy with the result, 
               change your prompts and/or the settings (CFG scale, random seed) and click "run inpaint" again.
    
            # EXAMPLES
            Scroll down to see a few examples. Click on an example and the image and the prompts will be filled for you. 
            Note however that you still need to do step 2 and 4.
            """)

        # State to store input points and original image
        state = gr.State(None)

        with gr.Row():
            # This is what the user will interact with
            display_img = gr.Image(
                label="Input",
                interactive=True,
                type='pil',
                height=IMG_SIZE,
                width=IMG_SIZE
            )

            sam_mask = gr.AnnotatedImage(
                label="SAM result",
                height=IMG_SIZE,
                width=IMG_SIZE,
                color_map={"background": "#a89a00"}
            )

            result = gr.Image(
                label="Output",
                interactive=False,
                type='pil',
                height=IMG_SIZE,
                width=IMG_SIZE,
            )

        with gr.Row():
            cfg = gr.Slider(
                label="Classifier-Free Guidance Scale",
                minimum=0.0,
                maximum=20.0,
                value=7,
                step=0.05
            )
            random_seed = gr.Number(
                label="Random seed",
                value=74294536,
                precision=0
            )
            checkbox = gr.Checkbox(
                label="Infill subject instead of background"
            )

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt for infill"
            )
            neg_prompt = gr.Textbox(
                label="Negative prompt"
            )

            reset_btn = gr.Button(value="Reset")
            submit_inpaint = gr.Button(value="Run inpaint", variant="primary")

        # Fixed event handlers
        # Handle image preprocessing on upload
        display_img.upload(
            fn=preprocess_image,
            inputs=[display_img],
            outputs=[display_img]
        )

        # Handle point selection
        display_img.select(
            fn=get_points,
            inputs=[display_img, state],
            outputs=[display_img, sam_mask, state]
        )

        # Handle reset
        reset_btn.click(
            fn=reset_state,
            outputs=[state],
            show_progress=False
        ).then(
            fn=lambda: (None, None, None),
            outputs=[display_img, sam_mask, result]
        )

        # Handle inpainting
        submit_inpaint.click(
            fn=run_inpaint,
            inputs=[state, prompt, neg_prompt, cfg, random_seed, checkbox],
            outputs=[result]
        )

        # Examples section
        examples = gr.Examples(
            [
                [
                    "car.png",
                    "a car driving on planet Mars. Studio lights, 1970s",
                    "artifacts, low quality, distortion",
                    74294536
                ],
                [
                    "dragon.jpeg",
                    "a dragon in a medieval village",
                    "artifacts, low quality, distortion",
                    97
                ],
                [
                    "monalisa.png",
                    "a fantasy landscape with flying dragons",
                    "artifacts, low quality, distortion",
                    97
                ]
            ],
            fn=load_example,
            inputs=[display_img, prompt, neg_prompt, random_seed],
            outputs=[display_img, prompt, neg_prompt, random_seed, state]
        )

    # Launch the demo
    demo.queue(max_size=1).launch(share=True, debug=True)

    return demo