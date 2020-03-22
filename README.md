# Place Recognition with ConvNet Landmarks
An object proposal techniques to identify potential landmarks within an
image for place recognition. We use the astonishing power
of convolutional neural network features to identify matching
landmark proposals between images to perform place recognition
over extreme appearance and viewpoint variations.

## Algorithm
1) landmark proposal extraction from the current image
2) calculation of a ConvNet feature for each proposal
3) projection of the features into a lower dimensional space
4) calculation of a matching score for each previously seen
image
5) calculation of the best match

![](https://github.com/sepidehhosseinzadeh/Visual-Place-Recognition/blob/master/pipeline.png)

## Notes
- Used https://github.com/pdollar/edges.git as edge-box shadow detection tool-box
- Used https://pdollar.github.io/toolbox/ as pdollar-toolbox

