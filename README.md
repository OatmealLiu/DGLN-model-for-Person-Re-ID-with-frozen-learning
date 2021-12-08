#Instructions

##0. Directory description
`condaEnv`: It contains the Conda environment of the project.\
`delivery`: It contains the deliveries for assignment task 1 and task 2.\
`Market`: It contains all the prepared datasets for the project.\
`master`: It contains all the code of the project.\
`master\feature`: It contains the extracted features of the images in `gallery_train`, `queries_train`, `gallery`, `queries` datasets.\
`master\model\ResNet50`: It contains the training logs and weights saved by every 10 epochs during training for multi-task classification network for person attributes prediction.\
`master\model_id\RGA_ResNet50`: It contains the training logs and weights saved by every 10 epochs during training for RGA network for person ID prediction.\
`weights\pre_train`: It contains the downloaded weights of ResNet-50 pretrained on Imagenet.\

##1. Environment setup
You can find the Conda environment in `condaEnv/dlbase_all_20210708.yml` folder.

##2. Train multi-task classification network for person attribute prediction
You can execute `train_attributes.py --gpu_ids 0 --name ResNet50 --data_dir ../Market --batchsize 32 --stride 2 --warm_epoch 10 --lr 0.01 --droprate 0.5 --num_epochs 60` to train the first model for person attribute prediction and local attribute feature extraction.

##3. Make the person attributes prediction for test dataset
You can execute `test_attributes.py--which_epoch 39` to make the prediction for test dataset. It will generate the `classification_test.csv` file for task 1 delivery.

##4. Train RGA model for person ID prediction
You can execute `train_personID.py --gpu_ids 0 --data_dir ../Market --batchsize 16 --stride 2 --warm_epoch 40 --optimizer Adam --lr 0.01 --wd 0.0005 --droprate 0.3 --num_epochs 240` to train the second model for person ID prediction and global feature extraction.

##5. Extract the features of gallery and queries dataset
You can execute `extractor.py --gpu_ids 0  --train_mode ON --batchsize 64 --name_local ResNet50 --name_global RGA_ResNet50 --which_epoch_local 39 --which_epoch_global 79` to extract the local and global features of the gallery_train and queries_train dataset.
You can execute `extractor.py --gpu_ids 0  --train_mode OFF --batchsize 64 --name_local ResNet50 --name_global RGA_ResNet50 --which_epoch_local 39 --which_epoch_global 79` to extract the local and global features of the gallery and queries dataset.

##6. Evaluate the mAP performance on gallery_train and queries_train dataset
You can execute `evaluator.py --make_pred OFF` to evalueate the mAP performance on gallery_train and queries_train dataset if you execute `extractor.py` with `--train_mode ON`
You can execute `evaluator.py --make_pred ON` to make and generate the re-id prediction result if you execute `extractor.py` with `--train_mode OFF`. It will generate `reid_test.txt` file for task 2 delivery.

##7. Visualize the top-10 Re-ID results on gallery_train and queries_train dataset
You can execute `visualization.py --query_index 66` to visualize the person re-id result of a specific query image, e.g. here we visualize the 66th image's top-10 re-id result.

