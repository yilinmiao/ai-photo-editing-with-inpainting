This project implements an AI-powered application for editing photos by replacing backgrounds with AI-generated content. The app uses:

1. **SAM (Segment Anything Model)** from Meta to automatically identify and segment objects in images
2. **Stable Diffusion XL** for inpainting to generate new backgrounds based on text prompts

## Project Files

- `AI_Photo_Editing_with_Inpainting.ipynb`: The main notebook where you'll complete the code
- `app.py`: The code for the interactive app that will be launched from the notebook
- `car.png`, `dragon.jpeg`, `monalisa.png`: Example images to test with

## Get Started

1. Open `AI_Photo_Editing_with_Inpainting.ipynb` in Google Colab (recommended) or a Jupyter notebook environment
2. Run all cells in the notebook to test the implementation
3. When running the app cell, use the public URL to interact with the app in a separate browser tab

Run the completed notebook to test if everything works correctly. The interactive app will allow you to:

1. Upload an image
2. Click on objects in the image to generate segmentation masks with SAM
3. Enter text prompts to describe the new background you want to generate
4. Generate new backgrounds using Stable Diffusion

## Troubleshooting

- Make sure all required libraries are installed (transformers, diffusers, etc.)
- The notebook is set up to use a GPU, make sure you have GPU acceleration enabled
