from Yolo import YoloSeg
from PIL import Image

# Load a model
model = YoloSeg('best.pt')  # load a custom model

# Predict with the model
results = model('processed_dataset/images/train/aachen_000002_000019_leftImg8bit.png')  # predict on an image

# Show the results
for r in results:
    im_array = r.plot(boxes=False)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image