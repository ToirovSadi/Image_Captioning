import warnings
warnings.filterwarnings("ignore")

model_names = {
    'transformer': 'models/transformer.ckpt',
    'seq2seq': 'models/seq2seq.ckpt',
    'seq2seq_1': 'models/seq2seq_1.ckpt',
}

def get_predict_fn(model_name):
    """
    Returns the predict function and checkpoint path based on the given model name.
    
    Args:
        model_name (str): Name of the model.
        
    Returns:
        tuple: A tuple containing the predict function and checkpoint path.
        
    Raises:
        ValueError: If the model name is incorrect.
    """
    if model_name not in model_names:
        raise ValueError(f"Incorrect model name, available one {list(model_names.keys())}")
    
    if model_name == 'transformer':
        from src.models.transformer.predict_model import predict
        return predict, model_names[model_name]
    if model_name == 'seq2seq' or model_name == 'seq2seq_1':
        from src.models.seq2seq.predict_model import predict
        return predict, model_names[model_name]


# import this function to predict, or you can predict running this python file and specifing the parameters
def predict(model_name: str, image_path: str, num_captions: int=1, max_steps: int=2000) -> list[str]:
    """
    Predicts captions for the given image using the specified model.
    
    Args:
        model_name (str): Name of the model to use for prediction.
        image_path (str): Path to the input image.
        num_captions (int, optional): Number of captions to generate. Defaults to 1.
        max_steps (int, optional): Maximum number of decoding steps. Defaults to 2000.
        
    Returns:
        list[str]: List of predicted captions.
    """
    predict_fn, ckpt_path = get_predict_fn(model_name)
    
    return predict_fn(ckpt_path, image_path, num_captions=num_captions, max_steps=max_steps)

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='transformer')
    p.add_argument('--image_path', type=str)
    p.add_argument('--num_captions', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=2000)
    args = p.parse_args()
    
    captions = predict(
        model_name=args.model_name,
        image_path=args.image_path,
        num_captions=args.num_captions,
        max_steps=args.max_steps,
    )
    
    if type(captions) is not list:
        captions = [captions]
    
    if len(captions) == 1:
        print("caption:", captions[0])
    else:
        print("captions:")
        for i, caption in enumerate(captions):
            print(f"{i}) {caption}")
