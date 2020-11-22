![pytorch.png](data/pytorch.png)

# [CaptionBot - Implementation of Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

README in progress. (Partially complete)

The [Caption Bot](https://share.streamlit.io/sankalp1999/image_captioning/main) takes your images and generates a caption in less than 40 words (even though a picture is worth a thousand words.....)

---

## Try the CaptionBot yourself!

Check out yourself [here](https://share.streamlit.io/sankalp1999/image_captioning/main).


You can check some of the results below.

The repository is divided into the following structure into three parts

1. [Pytorch model files](https://github.com/sankalp1999/Image_Captioning/tree/main/pytorch_model) - You can find the actual model files here. The .ipynb file(Kaggle kernel) on which the model was trained can be found in Kaggle notebook folder. 
2. Flask_App - this is inside main/pytorch_model. This was meant to deployed using flask. The requirements.txt here is specifically for deployment using Flask.
3. Main branch -  Deployment using streamlit.

You can download the weights from [Decoder_weights](https://www.dropbox.com/s/5ntq1bgp33k1197/LastModelResnet50_v2_16.pth.tar?dl=0) and [Encoder_Weights](https://www.dropbox.com/s/fot9zzgszkpsab7/resnet50_captioning.pt?dl=0).

Note: You can directly use resnet50(pretrained = True) because I did not fine-tune the resnet50.
Only the decoder is trained.

## How to run on local machine

1. Clone the repository

```python
git clone https://github.com/sankalp1999/Image_Captioning.git
cd sankalp1999/Image_Captioning.git
```

2. Make a new virtual environment (Optional)

```python
pip install --user virtualenv
```

You can change name from .env to something else.

```
virtualenv -p python3 .env
source .env/bin/activate
```

3. Install the required libraries. 

**Option 1** To train a similar model or see inference on Flask

```python
cd pytorch_model
pip install -r requirements.txt 
```

You will have to download the weights and keep it some proper path that you want to use.

Only inference

```python
# To try out just the inference, set image_path in file manually.
python inference.py

# For Flask app
cd Flask_App
python app.py
```

**Option 2**  Streamlit  Note that, this is the requirements.txt in the main branch. Additionally, you need to install wget command. 

```python
pip install -r requirements.txt
apt-get install -r packages.txt # Optional way to install wget
```

```python
# Note weights will get automatically downloaded. You have to set the right path
# in the streamlit_app file. If you have already downloaded, just place 
# in the path and make the changes in streamlit_app file also

streamlit run streamlit_app.py # Run the app on local machine

```

Additional note: The only difference between the requirements.txt files is streamlit.
Streamlit is a great platform which made the deployment part much easier. It's harder to deploy on Heroku.




## The Good
![1.png](/imgs/1.png)
---
![2.png](/imgs/2.png)
---
![3.png](/imgs/3.png)
---
![4.png](/imgs/4.png)
---


## Streamlit samples
![st1.png](/imgs/st1.png)
---
![st2.png](/imgs/st2.png)

