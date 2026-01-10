# Movie Recommendation System
Link cuộc thi trên Kaggle: https://www.kaggle.com/competitions/movie-recomendation-fall-2020/overview


# Cấu trúc Project
```
├───main.ipynb
├───README.md
├───data\
│   ├───test.txt
│   └───train.txt
├───models\
│   ├───attention_net.py
│   ├───dmf.py
│   ├───ensemble.py
│   ├───gmf.py
│   ├───lightgcn.py
│   ├───lightgcnpp.py
│   ├───ncf.py
│   ├───neumf.py
│   └───simgcl.py
├───pipeline\
│   ├───eval_model.py
│   ├───make_model.py
│   ├───pipeline.py
│   ├───preprocessing.py
│   └───train.py
└───util\
    ├───common.py
    └───submission_gen.py
```

## Mô tả cấu trúc
*   `data`: Chứa dữ liệu training và testing từ cuộc thi.
*   `models`: Chứa các file định nghĩa model.
*   `pipeline`: Chứa các file thực thi pipeline cho các model, bao gồm preprocessing -> training -> evaluation.
*   `util`: Chứa các hàm hỗ trợ chung và hàm generate ra submission file cho cuộc thi.

# Chạy mô hình
```
model_config = {...}
model = pipeline(model_name, model_config, preprocessing('./data/train.txt'))
```

## Ví dụ: GMF
```
gmf_config = {
    'epochs': 7,
    'batch_size': 512,
    'learning_rate': 0.00065,
    'weight_decay': 1e-5,
    'embedding_dim': 64,
    'device': 'cpu'
}
gmf = pipeline('gmf', gmf_config, preprocessing('./data/train.txt'))
```