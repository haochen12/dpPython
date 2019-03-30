from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import path

d = path.dirname(__file__)
text = open(path.join(d, "constitute.txt")).read()
alice_mask = np.array(Image.open(path.join(d, "timg.png")))
word_cloud = WordCloud(background_color='White', max_font_size=66, mask=alice_mask).generate(text)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
