for classifier in fat_classifier classifier; do
    export THEANO_FLAGS=compiledir=/tmp/theano.timing,device=gpu3,floatX=float32
    python -m rembed.models.${classifier} --eval_seq_length 50 --init_range 0.005 \
        --model_dim 300 --word_embedding_dim 300 \
        --data_type bl --learning_rate 0.0006 --batch_size 512 --training_steps 100 \
        --tracking_lstm_hidden_dim 34 --seq_length 50 --model_type Model0 \
        --l2_lambda 1.06503778953e-06 --experiment_name time_${classifier} \
        --nolstm_composition --nouse_tracking_lstm --noconnect_tracking_comp \
        --training_data_path ~/scr/tmp/pbl/pbl_train.tsv >> time_${classifier}.log 2>&1
done
