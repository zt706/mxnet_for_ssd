

# vgg16 reduce tupu face 开始训练
python train.py --train-path ./data_face/train.rec --val-path ./data_face/val.rec --gpus 3 --batch-size 8 --pretrained ./model/pretrain_model/vgg16_reduced --epoch 1 --prefix './model/new_train_model_tupu_face/vgg16_reduced/ssd' --network vgg16_reduced --data-shape 512 --class-names '0,1' --num-class 2

# vgg16 reduce 开始训练
python train.py --gpus 3 --batch-size 32 --pretrained ./model/pretrain_model/vgg16_reduced --epoch 1 --prefix './model/new_train_model/vgg16_reduced/ssd' --network vgg16_reduced --data-shape 512

# inceptionv3 tupu face 开始训练
python train.py --train-path ./data_face/train.rec --val-path ./data_face/val.rec --gpus 3 --batch-size 8 --pretrained ./model/pretrain_model/inceptionv3/Inception-7 --epoch 1 --prefix './model/new_train_model_tupu_face/inceptionv3/ssd' --network inceptionv3 --data-shape 300 --class-names '0,1' --num-class 2

# inceptionv3 开始训练
python train.py --gpus 0,3 --batch-size 32 --pretrained ./model/pretrain_model/inceptionv3/Inception-7 --epoch 1 --prefix './model/new_train_model/inceptionv3/ssd' --network inceptionv3 --data-shape 300 

# inceptionv3 继续训练
python train.py --gpus 0,3 --batch-size 32 --resume 25 --prefix './model/new_train_model/inceptionv3/ssd' --network inceptionv3 --data-shape 512

# inceptionv3 开始训练, 指定小数据集
python train.py --train-path ./data_voc_lit/train.rec --train-list ./data_voc_lit/train.lst --val-path  ./data_voc_lit/val.rec --val-list ./data_voc_lit/val.lst --gpus 0,3 --batch-size 32 --pretrained ./model/pretrain_model/inceptionv3/Inception-7 --epoch 1 --prefix './model/new_train_model/inceptionv3/ssd' --network inceptionv3 --data-shape 300 

