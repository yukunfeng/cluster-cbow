def load_word_emb(input_emb_path, emb_format):
    """
    emb_format: fasttext or word2vec. If word2vec format, bin or txt format will
    be automatically inferred from file name.
    """
    input_emb_path = os.path.expanduser(input_emb_path)
    # Load emb
    if args.emb_format == "fasttext":
        from gensim.models.wrappers import FastText
        emb_model = FastText.load_fasttext_format(input_emb_path)
    elif args.emb_format == "word2vec":
        from gensim.models import KeyedVectors
        if input_emb_path.endswith('bin'):
            binary = True
        elif input_emb_path.endswith('txt'):
            binary = False
        else:
            raise Exception(f"The binary type of {input_emb_path} is hard to infer.")
        emb_model = KeyedVectors.load_word2vec_format(input_emb_path, binary=binary)
    else:
        raise Exception(f"{args.emb_format} is not supported (only word2vec or fasttext).")

    return emb_model
