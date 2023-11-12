# HealthBuddy

Medical health chatbot built using BioBert & GPT-2 for medical queries.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies & Installations

_Prerequisites_:
* Python 3.8
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

### Executing program

_Training BioBert_
* Follow the steps in [Part1_BioBert_Finetuning_with_Question_Answer_Extractor_Models.ipynb](./Part1_BioBert_Finetuning_with_Question_Answer_Extractor_Models.ipynb)

_Training GPT-2_
* Follow the steps in [Part2_GPT2_Finetuning.ipynb](./Part2_GPT2_Finetuning.ipynb)

_Deploying the Inference Pipline & User Interface_
* Ensure that the pip packages in the [previous](#dependencies--installations) section are installed
* Run the following command in the terminal, it can take a few minutes to start (especially when running on CPU)
```sh
python deployment.py
```
* When the terminal displays a "finished initialization", the UI can be accessed via "http://localhost:42069" in the browser (port can be changed at the bottom of the [deployment.py](./deployment.py) file).

## Examples (images)

Examples can be found [here](./examples/README.md)

## Contributors

* [Hoon Qi Hang](#authors)
* [Yeo Zi Qing](#authors)
* [Edmund Chia](#authors)
* [Ranon Sew](#authors)

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
