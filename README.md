# arXiv summarizer
As an example of how to apply transformers in practice I wanted to make a few examples showcasing different tasks. This task being the summarization of a research paper, which is quite easy to do as there is a nice dataset available, as well as plenty of pretrained text summarization models.

Why would you want an arXiv summarizer? Indeed, research papers already come with an abstract so it would be kind of ridiculous to use this model instead. You can however also use this the other way around. When writing technical documents it is always interesting to have an executive summary for people without the time to go through the whole document. With this model you could quickly generate a first draft of such a summary, which you can then improve further. There is also another, less obvious, use for this kind of model: you can use it to check if your document is conveying your inteded message. When you are writing a technical document, you can let this kind of model generate a summary to see if your message is clear in the text you have written. If not, you might need to make some changes.

## What are transformers?
Not sure what transformers are or how they work? No worries I've got you covered with [this repository.](https://github.com/VerleysenNiels/transformers-pytorch)
Definitely also read the recommended papers to get a better understanding.

## A good standard practice
As you know transformers require a huge amount of data in order to attain a good performance. In practice, this can be quite a show-stopper. Huge datasets are expensive and hard to come by. And even then, there is a steep cost for training your model on this huge amount of data. 

It is therefore advised to start from a pretrained model on the same task (ideally on a similar dataset) and then finetune it on your own dataset. This lets you attain a much higher performance on your dataset that doesn't have to be huge, while consuming not as many resources. But, doesn't it take quite some time and effort to find, download and use a pretrained model? Not at all! Thanks to the platform [Hugging Face.](https://huggingface.co/) This platform provides you with everything you need, from pretrained models to datasets, documentation and even courses.

To find your model you go to the models tab, select the task for which you want a model. Maybe add some more tags, like the language you are working in or the deep learning framework you want to use. Once you have found a model, you can use their transformers library in python to easily load in the model. This final part is what these example projects are for.

## Environment
I have added the conda environment yaml to the repository, as well as a Dockerfile if you want to containerize this model. The most important two libraries are of course [Transformers](https://pypi.org/project/transformers/) and [PyTorch](https://pytorch.org/) with CUDA. I am running everything locally or on a GPU server in python, but you can always copy the code to a notebook and run it for instance in [Google Colab](https://colab.research.google.com/) if you don't have access to a GPU. 

## Dataset and model
For this specific repository I am using the [arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset that I found on the HuggingFace platform. As for the model, we need something that can efficiently process long sequences of tokens. The typical transformer cannot do this, as self-attention has a complexity that scales quadratically with sequence length. Where does this complexity come from? The attention mechanism for one token looks at all other tokens in the sequence, this is then repeated for all tokens in the sequence. In long pieces of text this can be a bit of overkill.. Most words in the text only relate to words that are somewhere near to it, for instance within the same paragraph or even within a part of that. So can't we put a limit on how far away the attention mechanism can look around a given token? In comes the [longformer](https://arxiv.org/pdf/2004.05150.pdf), which does exactly this and therefore reduces the complexity to be linear in sequence length. The longformer uses an attention pattern that slides over the text as attention is performed for each token in the sequence. You can think of this a bit like how a filter moves over an image in a convolution-layer. This is also what the researchers did, because they made clever use of a techniques from CNN's to further improve the efficiency. By using a dilated window, which is a window with holes in it, the attention mechanism can look at tokens a bit further away while keeping the performance in check.

That's all good and well, you might say, but what about references to ealier or later parts of text? You would be making a fair point. By only using a dilated sliding window, we are not accounting to references across the document. To deal with this problem a global attention part needs to be added into the mix. The longformer performs this with preselected locations across the text.

I'll be using the [led-base-16384](https://huggingface.co/allenai/led-base-16384) as starting point. This implementation can handle input sequences of up to 16k tokens, which should be enough for summarizing scientific papers.

## Training and results
As the inputs and outputs are rather large, training of the model was evidently going to take some time. For this reason I only trained the model on the arXiv dataset for a single epoch. This already took about 4 days on the GPU-server. I also reduced the number of allowed input tokens from 16384 to 10240 which should be enough for scientific papers. The training loss of this single epoch can be seen below.

![image](https://user-images.githubusercontent.com/26146888/222953350-e33f6b83-8e53-4385-a5c3-1d538cb5419b.png)

To test the model I first let it summarize the longformer paper for me. I excluded the abstract, as that would otherwise be a bit of cheating :wink:
Below are both the actual and the generated abstract from this paper. You can see which is which, but the generated abstract is not bad at all.

Generated:
> Long document transformers have achieved a state-of-the-art performance on a wide range of tasks, including classification, question answering ,and discriminative language understanding. We present a modified model that scales linearly with the sequence length, making it versatile for processing long documents on current long document transformer architectures. In particular, we show that our model outperforms the full self-attention operation of existing pretrained Transformers, and that it can be used to build contextual representations of the entire context using multiple layers of attention, reducing the need for task-specific architectures to address such interactions.

Real abstract:
> Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.

There is also a demo available .. TODO

## Bonus: Reddit TLDR model
I started this project with the idea to use a DistilBart model, however I forgot that these models don't really accept very long input sequences like for instance scientific publications. I didn't want to just throw everything away, so I finetuned this model on the reddit summarization dataset instead.

The reddit dataset can be found [here.](https://huggingface.co/datasets/reddit) The selected model is a [DistilBart-6-6 model](https://huggingface.co/sshleifer/distilbart-cnn-6-6) which was pretrained on the CNN news summarization dataset.

![Training loss over time](https://github.com/VerleysenNiels/arxiv-summarizer/blob/master/training_reddit/training_loss.png?raw=true)

Training the reddit model was unfortunately stopped due to a power outage... :scream: meaning that only the first epoch was completed. I then restarted the training which continued unhindered. You can see the training loss over time in the graph above. :point_up: In the meantime I went ahead and created a [small gradio demo](https://huggingface.co/spaces/NielsV/Reddit-TLDR-bot) to showcase this model. After only three days of training, this model already performs quite well. In the following example I took the top answer on reddit to [the question on how vanilla became the generic ice cream flavor](https://www.reddit.com/r/AskHistorians/comments/ijt3rd/how_did_vanilla_become_the_generic_flavor_of_ice/). The finetuned model summarizes the answer to:

> Vanilla was the first luxury flavor to be produced synthetically.
> Edit: spelling

Which is actually quite the decent answer. A funny thing is that there is an artifact from the training data in the generated response, the model added "Edit: spelling"  at the end (which is quite common on reddit).

![A gif visualizing the above example in the demonstrator](https://github.com/VerleysenNiels/arxiv-summarizer/blob/master/demo/reddit_demo.gif?raw=true)

Update: The model has finished training for three epochs. The model should perform better now, because of the two extra epochs. When observing the performance gain on the evaluation dataset, it is clear that in the current setup doing more epochs is not worth it anymore. The change between epochs two and three is already quite small. You can see the training and evaluation details on [HuggingFace](https://huggingface.co/NielsV/distilbart-cnn-6-6-reddit).

## Reusing trained models
I will publish the finetuned models on HuggingFace, so they can easily be reused.

[arXiv summarization model](https://huggingface.co/NielsV/led-arxiv-10240)

[Reddit TLDR model](https://huggingface.co/NielsV/distilbart-cnn-6-6-reddit)

## Other examples
This project is part of a bundle of three sideprojects focused on using transformers from HuggingFace in practice.

- [arXiv summarizer](https://github.com/VerleysenNiels/arxiv-summarizer) (this repository)
- [Image captioning](https://github.com/VerleysenNiels/image-captioning)
- [Historical map segmentation]() (coming soon)
