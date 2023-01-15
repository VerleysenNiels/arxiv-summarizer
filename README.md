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

For the model I settled with a [DistilBart-6-6 model](https://huggingface.co/sshleifer/distilbart-cnn-6-6) which was pretrained on the CNN news summarization dataset.

## Other examples
This project is part of a bundle of three sideprojects focused on using transformers from HuggingFace in practice.

- [arXiv summarizer](https://github.com/VerleysenNiels/arxiv-summarizer/edit/master/README.md) (this repository)
- [Image captioning](https://github.com/VerleysenNiels/image-captioning)
- [Historical map segmentation]() (coming soon)
