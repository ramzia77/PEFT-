from datasets import load_dataset

def get_dataset(name="tweet_eval", subset="sentiment"):
    """
    Loads Hugging Face dataset.
    Example: get_dataset("tweet_eval","sentiment")
    """
    return load_dataset(name, subset)

