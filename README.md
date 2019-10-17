Resources needed:

1) train set

.csv - names of all the training images and their corres true labels

image sources

2) test set

.csv - names, but no labels


verification code:

4/sAEnzv7-CuFAdcmJBwijLMZGplt4XdhcbvDnThNWB_DUTQ6diYxcaIY

Useful steps:

- for each icon, find around ~200 images; make sure all of them end with .JPG

- retrieve all file names into name.txt for all train set images:

ls -LR *.JPG > name.txt

- load all filenames into a .csv file for train set images

- add another column for images' labels (i.e. "5-polypropylene")

- upload images to colab:

https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92

> in my case, I used the 'pydrive'

- code to upload and unzip the image resources for train set:

```
!pip install PyDrive

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

download = drive.CreateFile({'id': '17glUfEAnIzIYujVOZytsCgNIxf2_Hwpm'})

download.GetContentFile('5_polypropylene_PP.zip')
!unzip 5_polypropylene_PP.zip
```

REF:

- image resources:
https://www.kaggle.com/piaoya/plastic-recycling-codes#seven_plastics_v9.zip

- format images and get file names for csv:
https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8

- all build steps:
https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/

- basic intro:
https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/


Extra Stuff:

- training from diff. types of files:
https://www.deepdetect.com/server/docs/csv-training/

- deploying model on GCP:
https://cloud.google.com/vision/automl/object-detection/docs/deploy


Other Training Cases:
- https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8
- https://towardsdatascience.com/fastai-image-classification-32d626da20
- https://towardsdatascience.com/https-medium-com-drchemlal-deep-learning-tutorial-1-f94156d79802

Transfer Learning Using Keras:
- https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
