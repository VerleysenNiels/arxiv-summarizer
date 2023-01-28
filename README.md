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
For this specific repository I am using the [arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset that I found on the HuggingFace platform.

## Bonus: Reddit TLDR model
I started this project with the idea to use a DistilBart model, however I forgot that these models don't really accept very long input sequences like for instance scientific publications. I didn't want to just throw everything away, so I finetuned this model on the reddit summarization dataset instead.

The reddit dataset can be found [here.](https://huggingface.co/datasets/reddit) The selected model is a [DistilBart-6-6 model](https://huggingface.co/sshleifer/distilbart-cnn-6-6) which was pretrained on the CNN news summarization dataset.

![Training loss over time](https://github.com/VerleysenNiels/arxiv-summarizer/blob/master/training_reddit/training_loss.png?raw=true)

Training the reddit model was unfortunately stopped due to a power outage... :scream: meaning that only the first epoch was completed. The training has been restarted and let's hope this time it can continue unhindered. You can see the training loss over time in the graph above. :point_up: In the meantime I went ahead and created a [small gradio demo](https://huggingface.co/spaces/NielsV/Reddit-TLDR-bot) to showcase this model. After only three days of training, this model already performs quite well. In the following example I took the top answer on reddit to [the question on how vanilla became the generic ice cream flavor](https://www.reddit.com/r/AskHistorians/comments/ijt3rd/how_did_vanilla_become_the_generic_flavor_of_ice/). The finetuned model summarizes the answer to:

> Vanilla was the first luxury flavor to be produced synthetically.
> Edit: spelling

Which is actually quite the decent answer. A funny thing is that there is an artifact from the training data in the generated response, the model added "Edit: spelling"  at the end (which is quite common on reddit).

![A gif visualizing the above example in the demonstrator](https://github.com/VerleysenNiels/arxiv-summarizer/blob/master/demo/reddit_demo.gif?raw=true)

## Reusing trained models
I will publish the finetuned models on HuggingFace, so they can easily be reused.

[Reddit TLDR model](https://huggingface.co/NielsV/distilbart-cnn-6-6-reddit)

I will add the arXiv model here when it is ready..

## Other examples
This project is part of a bundle of three sideprojects focused on using transformers from HuggingFace in practice.

- [arXiv summarizer](https://github.com/VerleysenNiels/arxiv-summarizer) (this repository)
- [Image captioning](https://github.com/VerleysenNiels/image-captioning)
- [Historical map segmentation]() (coming soon)
