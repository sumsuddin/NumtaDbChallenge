# NumtaDbChallenge

a. Description of the pipelines
-------------------------------

There are two scripts "TrainingCode" and "PredictionCode"

**TrainingCode :**
	
  1. At first there are the necessary imports and value initializations, followed by some utility methods.
	2. Overlay image generation code for train data augmentation. which tries to mimic the mirrored overlay images in the test set.
	3. Then the other data augmentation codes which includes rotation, shear, scale, translation, noise, dropouts, contrast changes etc. image processing.
	4. The codes that starts the data augmentation generation process.
	5. The rest of the code sets different parameters and runs the training process. Finally we get an optimized trained model which we will use to predict the labels.
	6. At the end there are ways by which we can save or load the model weights.

**PredictionCode :**
	
  1. Imports and initializations
	2. We used 3 different models which are trained similarly as the train code discussed.
	3. We used a method called learn.TTA() for our actual submissions. This can generate slightly different score depending on the random zoom of the images but it in turn improves the detection score. But it should result in scores appoximately around 0.992 - 0.993 accuracy in the overall test set. We predicted all the test images in a single script eg. no specific test dataset prediction.
	4. Finally writes the sorted data (based on key value) in csv file.


b. Description of the preprocessing
-----------------------------------

The preprocessing includes
	
  1. Overlay data generation: We used fixed 5000 [images](https://drive.google.com/open?id=1k_oD7gfcBiAjFlFoJZZjdva0D_nmWz5v) generated from the training data. We excluded 0 and 4 from being added as a mirrored image in the overlay. We used those 5000 images in all the training operations of the three models. But generating the overlay images by the provided script shouldn't change the final score significantly.
	2. The other data augmentation codes which includes rotation, shear, scale, translation, noise, dropouts, contrast changes etc. are standard image processing.
	3. The image is finally normalized for the used model by the built in "fastai" ModelData class.

**Augmentation Code**
```python
import imgaug as ia
from imgaug import augmenters as iaa

train_data_aug = iaa.Sequential(
    [
        iaa.SomeOf((1, 3),
            [
                iaa.Affine(
                    scale=(0.7, 1.1),
                    translate_percent={"x": (-0.1, 0.1), "y": (0.1, 0.1)},
                    rotate=(-30, 30),
                    shear=(-30, 30),
                    order=[0, 1],
                    cval=(0, 0),
                ),
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.15*255)
                    ),
                    iaa.SaltAndPepper(0.15),
                    iaa.Salt(0.15)
                ]),

                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05)),
                    iaa.CoarseDropout(
                        (0.03, 0.06), size_percent=(0.02, 0.04)
                    ),
                ]),

                iaa.OneOf([
                        iaa.OneOf([
                            iaa.GaussianBlur((3.0, 4.0)),
                            iaa.AverageBlur(k=(5, 7)),
                        ]),

                        iaa.Add((-10, 10), per_channel=0.5),
                    
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                ]),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
)
```

c. Description of the model used
--------------------------------
We used standart "ResNext" convolutional model without any change in layers except for finetuning the output layer using softmax activation. We used the pretrained weights from imagenet as the initial model weights. The learning rate used to update the model weights was 1e-3 and there were around 5 to 6 epochs for the actual training of the model.

As the training may require 2 to 3 hours of training time depending on the GPU used. I suggest, regenerate a single model and try replacing it with any of the provided models and see if the score remains approximately the same. Or of course if you want you can generate 2 or 3 models and take the votes for the prediction as stated in the script.
