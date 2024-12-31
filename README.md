The main reason why **ANN's are not preferred for processing images** :

**Firstly,**
ANN's treat each pixel as an individual features, which does not allow it to understand the spatial relations between them.
It completely ignores the spacial structure of the images, which plays a key role in tasks such as face detection, object recognition, etc.

**Secondly,**
There is no parameter sharing in ANN's.

**Thirdly,**
Number of parameters is large (especially for images where you have many pixels (1080p)) which makes computation more intense.

These are some major reasons why ANN's become incompetent to process images.
