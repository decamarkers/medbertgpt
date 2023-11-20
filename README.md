# HealthBuddy

Medical health chatbot built using BioBert & GPT-2 for medical queries.

## Description

CS425 Project: Healthbuddy. A chatbot system implemented with the aim to enable individuals to assess their symptoms and obtain preliminary medical advice, to ensure that they are able to receive timely and appropriate medical intervention as needed.

## Getting Started

### Dependencies & Installations

_Prerequisites_:
* Python 3.7^ (or newer)
* Jupyter Notebook & Conda (for training)
* Pip Packages (deployment.py)
```sh
# change "faiss-gpu" to "faiss-cpu" if GPU isn't enabled
pip install faiss-gpu tensorflow numpy pandas nltk transformers flask flask-cors
```

_Optional GPU drivers_
* [CUDA ToolKit](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/cudnn)
  * need to have Nvidia developer account
### How to run the code (Inference pipeline)
**Step 1**: Installed the required packages
```sh
# change "faiss-gpu" to "faiss-cpu" if GPU isn't enabled
pip install faiss-gpu tensorflow numpy pandas nltk transformers flask flask-cors
```

**Step 2**: Enter the directory
```sh
cd code
```

**Step 3**: Run the deployment file
```sh
python deployment.py
```

**Step 4**: When the terminal displays a "finished initialization" (2 mins), the UI can be accessed via [http://localhost:42069](http://localhost:42069)

### Fine Tune BioBert/GPT-2 Model

_Training BioBert_
* Follow the steps in [Part1_BioBert.ipynb](./Part_1_BioBert.ipynb)
  * Contains the data extraction and data cleaning, followed by the training of the BioBert model

_Training GPT-2_
* Follow the steps in [Part2_GPT2_Finetuning.ipynb](./Part2_GPT2_Finetuning.ipynb)
  * Contains the training of the GPT-2 decoder model

_Deploying the Inference Pipline & User Interface_
* Ensure that the pip packages in the [previous](#dependencies--installations) section are installed
* Run the following command in the terminal, it can take a few minutes to start (especially when running on CPU)
```sh
python deployment.py
```
* When the terminal displays a "finished initialization", the UI can be accessed via "http://localhost:42069" in the browser (port can be changed at the bottom of the [deployment.py](./deployment.py) file).
  * Do note that if using CPU to when running this script, it can take around 2 minutes (using a laptop 10th gen i5) to start the flask server, and can also take around 30 seconds (using the same laptop) to return a response when an input is provided

  * Certain browsers may block microphone access due the website being launched locally as "HTTP" instead of "HTTPS". To enable the microphone access you may have to whitelist the URL of the website as shown below in your browser
  * ![image](https://github.com/decamarkers/medbertgpt/assets/85857168/24e0031e-7d42-4621-9d1d-a66ca4b275be)

## Contributors

* Hoon Qi Hang
* Yeo Zi Qing
* Edmund Chia
* Ranon Sew
