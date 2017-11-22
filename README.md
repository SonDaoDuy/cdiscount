# cdiscount
kaggle cdiscount code

**TODO**

1. *segnet.py*:

	-loss function

2. *segnet_train.py*:

	-save model


3. *data_manipulate.py*:

	-loaddata into train set and test set

	-split the data into binary class

	-other manipulate data function if needed 



4. *segnet_predict.py*:

	-predict testing data

5. train cdiscount by Inception Resnet v2
	-python inception_resnet_v2.py

6. train cdiscount by Inception Resnet v2 with horovod
	mpirun -np 2 \
    -H localhost:2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    python inception_resnet_v2_with_horovod.py

    Các tham số được mô tả ở https://github.com/uber/horovod/blob/master/docs/running.md