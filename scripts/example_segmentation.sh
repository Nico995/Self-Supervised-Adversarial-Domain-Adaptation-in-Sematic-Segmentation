python adaptation.py --num_epochs 50 --batch_size 2 --learning_rate 0.025 --optimizer sgd --loss dice --checkpoint_step 20 \
 --validation_step 10 --crop_height 720 --crop_width 960 --context_path resnet18 --num_classes 12 --data ./data/CamVid/ \
 --save_model_path ./checkpoints --num_workers 6 --use_gpu --pre_encoded