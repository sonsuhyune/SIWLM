# The Sequential and Intensive Weighted Language Modeling for Natural Language Understanding


Note that we used three V100 GPUs and the same hyperparameter reported on [MT-DNN](https://arxiv.org/abs/1901.11504) paper for pre-training: a learning rate of 5e-5, a batch size of 32 and an optimizer of Adamax. 

Also, for the fine-tuning stage, we followed the hyperparameter range suggested by [SMART](https://arxiv.org/abs/1911.03437) paper: a learning rate of {1e-5, 2e-5, 3e-5, 5e-5\}, a batch size of {16, 32, 64\} and an optimizer of Adam.


<br/>


### Setup Environment

1. python 3.6

   Reference to download and install : https://www.python.org/downloads/release/python-360/
   
<br/>   

2. install requirements

   ```> pip install -r requirements.txt```

<br/>



### Train a SIWLM model

1. **Download data**

   ```> sh download.sh``` 

   Please refer to download GLUE dataset: https://gluebenchmark.com/

     

2. **Preprocess data**

   ```> sh experiments/glue/prepro.sh```

     

3.  **Set the task weight.**

      At *experiments/glue/glue_task_def.yml*, you can set the weight of each task.

      We set the initial task weights as \{1:1:1:1:1:1:1:1, 3:1:1:1:1:1:1:1, 6:1:1:1:1:1:1:1, 9:1:1:1:1:1:1:1, 12:1:1:1: 1:1:1:1, 15:1:1:1:1:1:1:1\}. The first values in the task     weights are for a central task.

   

4. **Multi-task learning with the model with Sequential and Intensive Weighted language modeling method**

   Using scripts,

   ```> sh scripts/run_mtdnn_{task_name}.sh {batch_size}```

   We provide example, MRPC. You can use similar scripts to train other GLUE tasks.

   : ```> sh scripts/run_mtdnn_mrpc.sh 32```



5. **Strip the task-specific layers  for fine-tuning**

   ```> python strip_model.py  --model_path {multi-task learned model path} --fout {striped model path}```



6. **Fine-tuning**

   ```> sh scripts/run_{task_name}.sh {batch_size}```

   We provide example, MRPC. You can use similar scripts to fine-tune other GLUE tasks.

   ```> sh scripts/run_mrpc.sh 32```

   

### Codebase

MT-DNN repo : [https://github.com/namisan/mt-dnn](https://github.com/namisan/mt-dnn)



### Contact

Email
