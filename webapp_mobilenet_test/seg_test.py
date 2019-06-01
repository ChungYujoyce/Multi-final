import image_test

# Dataset names.
_ADE20K = 'ade20k'
_CITYSCAPES = 'cityscapes'
_MAPILLARY_VISTAS = 'mapillary_vistas'
_PASCAL = 'pascal'

#Model names
_Cityscapes_mobilenet_stride16 = 'models/mobilev2_restride16_100000.tar.gz'
_Pascal_deeplab = 'models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz'
_Cityscapes_mobilenet = 'models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
_Pascal_mobilenet = 'models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
image_test.image_inference(_Pascal_mobilenet, 'models/dog_person.jpg', _PASCAL)

