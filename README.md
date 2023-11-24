# Image_Captioning

## Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. Most image captioning systems use an encoder-decoder framework, where an input image is encoded into an intermediate representation of the information in the image, and then decoded into a descriptive text sequence.


# How to use
## How to install
```bash
git clone https://github.com/ToirovSadi/Image_Captioning.git

cd Image_Captioning
pip install -r requirements.txt
```

Install all requirements for the model, to avoid any errors. It's recommented to create a new python env before installing all those packages.


**Note:** Please install model that you want to use before predicting and put it in models/
## How to predict
```bash
python predict.py --model_name='transformer' --image_path='/path/to/image'
```

Resplace the `/path/to/image` with your image path, and run the script, it will run our image through the model and make prediction. It will output a string.

### Model Names
- transformer
- transformer_small
- seq2seq

I recommend using `transformer`. If you want to make several predictions for a single image then pass `--num_captions=N` to predict `N` captions, for now works only with `transformer` and `transformer_small`.

See [demo](./demo.ipynb) to understant how to make predictions.
