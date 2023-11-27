from wordcloud import WordCloud
from bertopic import BERTopic
import matplotlib.pyplot as plt


def create_wordcloud_from_topic(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    return wc


def display_topic_histogram(trained_bert: BERTopic):
    trained_bert.visualize_barchart()
    