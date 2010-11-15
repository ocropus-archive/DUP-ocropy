from pylab import *
import ocropy
from ocropy import dbtables  

ocropy.dinit(512,512)

ion()
show()

# Let's open a character database and read some characters.

table = dbtables.Table("test2.db","clusters")
table.converter("image",dbtables.SmallImage())
clusters = table.get()

shapes = []
for c in clusters:
    # pick one of the characters
    image = c.image 
    subplot(331); imshow(image,interpolation='nearest',cmap=cm.gray); draw()

    # allocate the feature extractor
    extractor = ocropy.make_IExtractor("StandardExtractor")

    # allocate an array of floatarrays to hold the feature maps
    featuremap = ocropy.floatarrayarray()

    # now perform the feature extraction
    extractor.extract(featuremap,ocropy.FI(image))

    # finally, let's look at the individual maps
    out = ocropy.floatarray()

    # horizontal gradients; these alternate between positive and 
    # negative parts for each row
    featuremap.get(out,0)
    subplot(332); imshow(ocropy.NI(out),interpolation='nearest'); draw()

    # vertical gradients; these again alternate between 
    # positive and negative parts for each column
    featuremap.get(out,1)
    subplot(333); imshow(ocropy.NI(out),interpolation='nearest'); draw()

    # junctions, endpoints, and holes
    featuremap.get(out,2)
    subplot(334); imshow(ocropy.NI(out),interpolation='nearest',cmap=cm.gray,vmin=0.0,vmax=0.3); draw()

    featuremap.get(out,3)
    subplot(335); imshow(ocropy.NI(out),interpolation='nearest',cmap=cm.gray,vmin=0.0,vmax=0.3); draw()

    featuremap.get(out,4)
    subplot(336); imshow(ocropy.NI(out),interpolation='nearest',cmap=cm.gray,vmin=0.0,vmax=0.3); draw()
    raw_input()

