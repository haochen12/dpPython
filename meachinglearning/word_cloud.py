from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud.generate_from_text(['指环王2：双塔奇兵'], frequencies=[1])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
